"""
tests/test_app.py
Comprehensive pytest suite for CrisisBridge Emergency Response Hub.

Tests:
1. Health check and monitoring endpoints.
2. Static dashboard accessibility and landing page delivery.
3. Strict Pydantic input validation and boundary checks.
4. Two-stage Google Gemini 2.5 Flash pipeline with mocked AI client.
5. Resilient fallback engine across various disaster categories.
6. Ticket listing, search filtering, and status progression workflows.
7. System-wide statistics computation and confidence score boundaries.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app, INCIDENT_DATABASE, seed_initial_tickets
from services.gemini_service import (
    disaster_engine,
    GeminiDisasterEngine,
    TriageResult,
    VerificationResult,
    IncidentType,
    UrgencyLevel,
    TicketStatus
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    """Ensure database has baseline seed data before tests run."""
    seed_initial_tickets()
    yield


# ----------------------------------------------------------------------
# 1. Health and Root Dashboard Tests
# ----------------------------------------------------------------------

def test_health_check():
    """Verify health check endpoint returns 200 and required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CrisisBridge" in data["service"]
    assert "uptime_seconds" in data
    assert "gemini_api_configured" in data
    assert data["total_tickets_tracked"] >= 1


def test_api_health_alias():
    """Verify /api/health alias functions identically."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_dashboard():
    """Verify GET / returns accessible HTML dashboard with CrisisBridge title."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CRISIS" in response.text
    assert "Victim Signal Simulator" in response.text


# ----------------------------------------------------------------------
# 2. Strict Input Validation Tests
# ----------------------------------------------------------------------

def test_input_validation_empty_payload():
    """Reject request with missing body."""
    response = client.post("/api/triage", json={})
    assert response.status_code == 422


def test_input_validation_empty_text():
    """Reject empty distress text."""
    response = client.post("/api/triage", json={"raw_text": ""})
    assert response.status_code == 422


def test_input_validation_whitespace_only():
    """Reject whitespace-only signal (custom validator)."""
    response = client.post("/api/triage", json={"raw_text": "        "})
    assert response.status_code == 422


def test_input_validation_too_short():
    """Reject signals under 5 characters."""
    response = client.post("/api/triage", json={"raw_text": "SOS!"})
    assert response.status_code == 422


def test_input_validation_excessive_length():
    """Reject signal exceeding max length (4000 characters)."""
    long_text = "Emergency! " * 450
    response = client.post("/api/triage", json={"raw_text": long_text})
    assert response.status_code == 422


# ----------------------------------------------------------------------
# 3. Two-Stage Triage Endpoint & Flow Tests
# ----------------------------------------------------------------------

def test_triage_submission_end_to_end():
    """Verify submitting a valid emergency signal creates a verified ticket."""
    payload = {
        "raw_text": "Flash flood breached the river bank. 3 people stuck on car roof at 100 Feet Road, Indiranagar near KFC. Water rising fast!",
        "source": "SMS SOS",
        "reporter_name": "Arjun S.",
        "contact_info": "+91 99000 11223"
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 201
    ticket = response.json()

    assert ticket["ticket_id"].startswith("CRB-")
    assert ticket["source"] == "SMS SOS"
    assert ticket["reporter_name"] == "Arjun S."

    # Stage 1 Triage assertions
    triage = ticket["triage"]
    assert triage["incident_type"] == "Flood"
    assert triage["urgency_level"] in ["CRITICAL", "HIGH"]
    assert triage["victims_trapped"] >= 3
    assert "Indiranagar" in triage["location"] or "Road" in triage["location"]
    assert len(triage["required_resources"]) > 0

    # Stage 2 Verification assertions
    verification = ticket["verification"]
    assert 0 <= verification["confidence_score"] <= 100
    assert isinstance(verification["verified"], bool)
    assert len(verification["verification_reasoning"]) > 5


def test_mocked_gemini_two_stage_pipeline():
    """
    Test that the GeminiDisasterEngine coordinates Agent 1 and Agent 2
    when google-genai client returns structured outputs.
    """
    mock_triage_result = TriageResult(
        incident_type=IncidentType.FIRE,
        urgency_level=UrgencyLevel.CRITICAL,
        victims_trapped=4,
        location="Unit 3, Peenya Industrial Complex",
        required_resources=["Fire Ladder", "Breathing Apparatus (SCBA)"],
        summary="Severe chemical warehouse fire with 4 trapped workers.",
        hazards_detected=["Dense smoke", "Flammable solvent barrels"]
    )

    mock_verification_result = VerificationResult(
        confidence_score=95,
        verified=True,
        urgency_confirmation=UrgencyLevel.CRITICAL,
        hallucination_detected=False,
        verification_reasoning="Report directly matches source radio call. Location and 4 trapped workers corroborated.",
        flagged_inconsistencies=[],
        recommended_action="Dispatch ladder unit and Hazmat squad immediately."
    )

    # Mock google-genai Client responses
    mock_genai_client = MagicMock()
    
    mock_resp1 = MagicMock()
    mock_resp1.parsed = mock_triage_result
    mock_resp1.text = mock_triage_result.model_dump_json()

    mock_resp2 = MagicMock()
    mock_resp2.parsed = mock_verification_result
    mock_resp2.text = mock_verification_result.model_dump_json()

    # Agent 1 called first, then Agent 2
    mock_genai_client.models.generate_content.side_effect = [mock_resp1, mock_resp2]

    test_engine = GeminiDisasterEngine(api_key="mock_api_key_test")
    test_engine._client = mock_genai_client

    import asyncio
    ticket = asyncio.run(test_engine.process_signal(
        raw_text="Chemical warehouse fire at Unit 3 Peenya. 4 workers trapped inside!",
        source="Radio Call"
    ))

    assert ticket.triage.incident_type == IncidentType.FIRE
    assert ticket.triage.victims_trapped == 4
    assert ticket.verification.confidence_score == 95
    assert ticket.verification.verified is True
    assert mock_genai_client.models.generate_content.call_count == 2


# ----------------------------------------------------------------------
# 4. Ticket Queries, Filtering, and Status Update Tests
# ----------------------------------------------------------------------

def test_get_tickets_listing():
    """Verify tickets can be queried and are sorted by urgency."""
    response = client.get("/api/tickets")
    assert response.status_code == 200
    tickets = response.json()
    assert isinstance(tickets, list)
    assert len(tickets) >= 1

    # Check priority sorting: CRITICAL should be first
    urgencies = [t["triage"]["urgency_level"] for t in tickets]
    if "CRITICAL" in urgencies and "LOW" in urgencies:
        crit_index = urgencies.index("CRITICAL")
        low_index = urgencies.index("LOW")
        assert crit_index < low_index


def test_get_tickets_filter_urgency():
    """Verify filtering by urgency returns only matched items."""
    response = client.get("/api/tickets?urgency=CRITICAL")
    assert response.status_code == 200
    tickets = response.json()
    for t in tickets:
        assert t["triage"]["urgency_level"] == "CRITICAL"


def test_get_tickets_search():
    """Verify keyword search finds specific location or words."""
    response = client.get("/api/tickets?search=Peenya")
    assert response.status_code == 200
    tickets = response.json()
    for t in tickets:
        combined = (t["triage"]["location"] + t["raw_text"] + t["triage"]["summary"]).lower()
        assert "peenya" in combined


def test_get_single_ticket():
    """Verify fetching single ticket by ID."""
    all_tickets = client.get("/api/tickets").json()
    target_id = all_tickets[0]["ticket_id"]

    response = client.get(f"/api/tickets/{target_id}")
    assert response.status_code == 200
    assert response.json()["ticket_id"] == target_id


def test_get_single_ticket_not_found():
    """Verify 404 returned for unknown ticket ID."""
    response = client.get("/api/tickets/CRB-NONEXISTENT-999")
    assert response.status_code == 404


def test_update_ticket_status():
    """Verify status can be transitioned from PENDING -> DISPATCHED -> RESOLVED."""
    all_tickets = client.get("/api/tickets").json()
    target_id = all_tickets[0]["ticket_id"]

    # Transition to DISPATCHED
    res1 = client.patch(f"/api/tickets/{target_id}/status", json={"status": "DISPATCHED"})
    assert res1.status_code == 200
    assert res1.json()["status"] == "DISPATCHED"

    # Transition to RESOLVED
    res2 = client.patch(f"/api/tickets/{target_id}/status", json={"status": "RESOLVED"})
    assert res2.status_code == 200
    assert res2.json()["status"] == "RESOLVED"


def test_update_ticket_status_invalid_status():
    """Verify invalid status string returns 422."""
    all_tickets = client.get("/api/tickets").json()
    target_id = all_tickets[0]["ticket_id"]

    res = client.patch(f"/api/tickets/{target_id}/status", json={"status": "INVALID_STATE"})
    assert res.status_code == 422


# ----------------------------------------------------------------------
# 5. Statistics and Metric Aggregation Tests
# ----------------------------------------------------------------------

def test_system_stats():
    """Verify /api/stats accurately aggregates metrics."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    stats = response.json()

    assert stats["total_tickets"] >= 1
    assert stats["critical_tickets"] >= 0
    assert stats["total_victims_trapped"] >= 0
    assert 0 <= stats["average_confidence_score"] <= 100
    assert isinstance(stats["gemini_ai_configured"], bool)


# ----------------------------------------------------------------------
# 6. Fallback Heuristic Multi-Disaster Classification Tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_incident,expected_urgency", [
    (
        "Water rising fast, river flooded the road, 2 people on roof help!",
        IncidentType.FLOOD,
        UrgencyLevel.CRITICAL
    ),
    (
        "Huge fire and black smoke spreading through the apartment building, flame is uncontrollable!",
        IncidentType.FIRE,
        UrgencyLevel.CRITICAL
    ),
    (
        "Patient having severe chest pain, cardiac arrest suspected, unconscious on floor!",
        IncidentType.MEDICAL,
        UrgencyLevel.CRITICAL
    ),
    (
        "Earthquake rubble collapsed the ceiling, 3 workers crushed under debris!",
        IncidentType.STRUCTURAL,
        UrgencyLevel.CRITICAL
    )
])
def test_heuristic_triage_accuracy(text, expected_incident, expected_urgency):
    """Ensure deterministic fallback engine reliably maps distress keywords."""
    result = disaster_engine._heuristic_triage(text)
    assert result.incident_type == expected_incident
    assert result.urgency_level == expected_urgency
    assert len(result.required_resources) > 0


def test_ticket_filter_by_incident_type():
    """Verify filtering by incident_type returns only matching incidents."""
    response = client.get("/api/tickets?incident_type=Flood")
    assert response.status_code == 200
    tickets = response.json()
    for t in tickets:
        assert t["triage"]["incident_type"] == "Flood"


def test_ticket_filter_by_status():
    """Verify filtering by dispatch status."""
    response = client.get("/api/tickets?status=PENDING")
    assert response.status_code == 200
    tickets = response.json()
    for t in tickets:
        assert t["status"] == "PENDING"


def test_engine_resilience_when_gemini_fails():
    """Verify that when the Gemini client raises an exception, the fallback catches it."""
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("Quota exceeded or network timeout")

    engine = GeminiDisasterEngine(api_key="mock_key")
    engine._client = failing_client

    import asyncio
    ticket = asyncio.run(engine.process_signal(
        raw_text="Submerged house, 4 elders trapped on roof at Indira Nagar",
        source="SMS SOS"
    ))

    # Should not crash, returns valid ticket with fallback
    assert ticket.triage.incident_type == IncidentType.FLOOD
    assert ticket.triage.victims_trapped >= 4
    assert ticket.verification.verified is True
    assert ticket.verification.confidence_score >= 50

