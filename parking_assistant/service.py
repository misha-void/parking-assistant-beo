"""
Core chat service - integrates RAG chain and reservation slot collection.
"""

from typing import List, Dict, Optional
from datetime import datetime

from parking_assistant.llm_factory import get_chat_model
from parking_assistant.rag.chain import RAGChain
from parking_assistant.graph.slot_collector import (
    SlotCollector, 
    ReservationSlots, 
    parse_datetime_with_llm,
    match_location_with_llm
)
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import Reservation, ReservationStatus, ParkingLocation
from parking_assistant.admin.admin_agent import escalate_to_admin


# In-memory session state storage
# In production, use Redis or similar
SESSION_STATES = {}


class ConversationState:
    """Tracks conversation state for a user session."""
    
    def __init__(self):
        self.mode = "chat"  # "chat" or "reservation"
        self.reservation_slots = ReservationSlots()
        self.awaiting_confirmation = False
        self.last_asked_slot = None
    
    def reset(self):
        """Reset to initial state."""
        self.mode = "chat"
        self.reservation_slots = ReservationSlots()
        self.awaiting_confirmation = False
        self.last_asked_slot = None


async def process_chat_message(
    user_message: str,
    session_id: str = "default",
    chat_history: List[Dict[str, str]] = None
) -> str:
    """
    Process a user message with RAG + reservation handling.
    
    Args:
        user_message: The user's input message
        session_id: Unique session identifier for state tracking
        chat_history: Optional list of previous messages (not used for now)
    
    Returns:
        The assistant's response as a string
    """
    llm = get_chat_model(temperature=0.7)
    
    # Get or create session state
    if session_id not in SESSION_STATES:
        SESSION_STATES[session_id] = ConversationState()
    
    state = SESSION_STATES[session_id]
    
    # Handle reservation mode
    if state.mode == "reservation":
        return await handle_reservation_mode(user_message, state, llm)
    
    # Normal chat mode - use RAG chain
    rag_chain = RAGChain(llm)
    
    try:
        result = await rag_chain.run(user_message)
        
        # Check if user wants to book
        if result["intent"] == "book_spot":
            state.mode = "reservation"
            state.reservation_slots = ReservationSlots()
            
            return ("Great! I'll help you book a parking spot. "
                    "Let me collect some information.\n\n"
                    "What's your first name?")
        
        return result["response"]
    
    finally:
        rag_chain.close()


async def handle_reservation_mode(
    user_message: str,
    state: ConversationState,
    llm
) -> str:
    """Handle slot-filling conversation for reservations."""
    
    collector = SlotCollector()
    message_lower = user_message.lower().strip()
    
    # Check for cancellation
    if message_lower in ["cancel", "stop", "quit", "nevermind"]:
        state.reset()
        return "Reservation cancelled. How else can I help you?"
    
    # If awaiting confirmation
    if state.awaiting_confirmation:
        if message_lower in ["yes", "confirm", "correct", "ok"]:
            # Save reservation to database
            reservation_id = await save_reservation(state.reservation_slots)

            # Escalate to human administrator (second agent)
            await escalate_to_admin(reservation_id, llm)

            state.reset()  # Reset state after saving

            return (
                f"✅ **Reservation Submitted!**\n\n"
                f"Your reservation request (ID: #{reservation_id}) has been submitted for approval.\n"
                f"A staff member will review and confirm your booking shortly.\n\n"
                f"You'll receive confirmation once approved. Thank you!"
            )
        else:
            state.reset()
            return "Reservation cancelled. Let me know if you'd like to try again!"
    
    # Fill slots based on what we're asking
    slots = state.reservation_slots
    
    if not slots.name:
        slots.name = user_message.strip()
        state.last_asked_slot = "name"
        return collector.SLOT_PROMPTS["surname"]
    
    elif not slots.surname:
        slots.surname = user_message.strip()
        state.last_asked_slot = "surname"
        return collector.SLOT_PROMPTS["car_number"]
    
    elif not slots.car_number:
        car_number = user_message.strip().upper()
        if collector.validate_car_number(car_number):
            slots.car_number = car_number
            state.last_asked_slot = "car_number"
            
            # Show available locations
            session = SessionLocal()
            try:
                locations = session.query(ParkingLocation).limit(5).all()
                location_list = "\n".join([
                    f"{i+1}. {loc.name} - {loc.address}"
                    for i, loc in enumerate(locations)
                ])
                
                return (
                    f"Here are some available parking locations:\n\n{location_list}\n\n"
                    f"Which location would you prefer? (type the number or name)"
                )
            finally:
                session.close()
        else:
            return "Invalid car number format. Please enter a valid license plate (e.g., BG123AB):"
    
    elif not slots.location_id:
        # Try to match location
        session = SessionLocal()
        try:
            locations = session.query(ParkingLocation).all()
            
            # Check if user typed a number
            try:
                choice = int(user_message.strip()) - 1
                if 0 <= choice < len(locations):
                    slots.location_id = locations[choice].id
                    slots.location_preference = locations[choice].name
            except ValueError:
                # Use LLM to match description
                loc_list = [
                    {"id": loc.id, "name": loc.name, "address": loc.address}
                    for loc in locations
                ]
                matched_id = await match_location_with_llm(user_message, loc_list, llm)
                
                if matched_id:
                    slots.location_id = matched_id
                    matched_loc = next(l for l in locations if l.id == matched_id)
                    slots.location_preference = matched_loc.name
            
            if slots.location_id:
                state.last_asked_slot = "location"
                return collector.SLOT_PROMPTS["start_date"]
            else:
                return "I couldn't match that to a location. Please choose a number from the list above:"
        finally:
            session.close()
    
    elif not slots.start_date:
        # Parse start date
        parsed = await parse_datetime_with_llm(user_message, llm)
        if parsed:
            slots.start_date = parsed.isoformat()
            state.last_asked_slot = "start_date"
            return collector.SLOT_PROMPTS["end_date"]
        else:
            return "I couldn't understand that date/time. Please try again (e.g., 'tomorrow at 9am', 'Dec 25 at 10:00'):"
    
    elif not slots.end_date:
        # Parse end date
        parsed = await parse_datetime_with_llm(user_message, llm)
        if parsed:
            slots.end_date = parsed.isoformat()
            state.last_asked_slot = "end_date"
            
            # All slots filled - show confirmation
            session = SessionLocal()
            try:
                location = session.query(ParkingLocation).filter(
                    ParkingLocation.id == slots.location_id
                ).first()
                
                confirmation = collector.format_confirmation(slots, location.name)
                state.awaiting_confirmation = True
                
                return confirmation
            finally:
                session.close()
        else:
            return "I couldn't understand that date/time. Please try again (e.g., 'same day at 5pm', 'Dec 25 at 18:00'):"
    
    # Shouldn't reach here
    return "Something went wrong. Let's start over. Type 'cancel' to exit."


async def save_reservation(slots: ReservationSlots) -> int:
    """Save reservation to database."""
    session = SessionLocal()
    try:
        reservation = Reservation(
            user_name=slots.name,
            user_surname=slots.surname,
            car_number=slots.car_number,
            location_id=slots.location_id,
            start_datetime=datetime.fromisoformat(slots.start_date),
            end_datetime=datetime.fromisoformat(slots.end_date),
            status=ReservationStatus.PENDING_APPROVAL
        )
        
        session.add(reservation)
        session.commit()
        session.refresh(reservation)
        
        return reservation.id
    finally:
        session.close()
