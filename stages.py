import json
import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from mcp_client import common_client, atlas_client

logger = logging.getLogger(__name__)

# ============= State Management =============
class WorkflowState(TypedDict):
    """State that persists across all stages"""
    # Initial payload
    customer_name: str
    email: str
    query: str
    priority: str
    ticket_id: str
    
    # Stage 2: Understanding
    parsed_intent: Optional[str]
    entities: Optional[Dict]
    sentiment: Optional[str]
    
    # Stage 3: Preparation
    normalized_fields: Optional[Dict]
    enriched_data: Optional[Dict]
    flags_calculations: Optional[Dict]
    
    # Stage 4-5: Ask & Wait
    clarification_needed: Optional[bool]
    clarification_question: Optional[str]
    customer_response: Optional[str]
    
    # Stage 6: Retrieve
    knowledge_base_results: Optional[List[Dict]]
    relevant_data: Optional[Dict]
    
    # Stage 7: Decide
    solutions: Optional[List[Dict]]
    chosen_solution: Optional[Dict]
    escalation_required: Optional[bool]
    
    # Stage 8-9: Update & Create
    ticket_status: Optional[str]
    generated_response: Optional[str]
    
    # Stage 10: Do
    executed_actions: Optional[List[str]]
    notifications_sent: Optional[List[str]]
    
    # Stage 11: Complete
    final_status: Optional[str]
    processing_time: Optional[str]
    
    # Metadata
    stage_history: List[str]
    current_stage: str
    timestamp: str


# ============= Stage Implementations =============

async def stage_1_intake(state: WorkflowState) -> WorkflowState:
    """Stage 1: INTAKE - Accept initial payload"""
    logger.info("🔵 Stage 1: INTAKE")
    
    # Validate required fields
    required_fields = ['customer_name', 'query', 'ticket_id']
    for field in required_fields:
        if not state.get(field):
            state[field] = ""
    
    state['current_stage'] = 'INTAKE'
    state['stage_history'].append('INTAKE')
    state['timestamp'] = datetime.now().isoformat()
    
    logger.info(f"✅ Payload accepted: Ticket {state['ticket_id']}")
    return state

async def stage_2_understand(state: WorkflowState) -> WorkflowState:
    """Stage 2: UNDERSTAND - Parse and extract entities (Deterministic)"""
    logger.info("🔵 Stage 2: UNDERSTAND (Deterministic)")
    
    # Parse request text with enhanced AI (COMMON)
    parsed = await common_client.call_ability("parse_request_text", {
        "text": state['query']
    })
    state['parsed_intent'] = parsed.get('intent')
    state['sentiment'] = parsed.get('sentiment')
    
    # Store additional parsed information
    state['urgency'] = parsed.get('urgency', 'medium')
    state['complexity'] = parsed.get('complexity', 'moderate')
    state['customer_emotion'] = parsed.get('customer_emotion', 'calm')
    
    # Extract entities (ATLAS)
    entities = await atlas_client.call_ability("extract_entities", {
        "text": state['query']
    })
    state['entities'] = entities.get('entities', {})
    
    state['current_stage'] = 'UNDERSTAND'
    state['stage_history'].append('UNDERSTAND')
    
    logger.info(f"✅ Intent: {state['parsed_intent']}, Sentiment: {state['sentiment']}, Urgency: {state['urgency']}")
    logger.info(f"✅ Entities: {state['entities']}")
    return state

async def stage_3_prepare(state: WorkflowState) -> WorkflowState:
    """Stage 3: PREPARE - Normalize and enrich data (Deterministic)"""
    logger.info("🔵 Stage 3: PREPARE (Deterministic)")
    
    # Normalize fields (COMMON)
    normalized = await common_client.call_ability("normalize_fields", {
        "data": state['entities']
    })
    state['normalized_fields'] = normalized.get('normalized', {})
    
    # Enrich records (ATLAS)
    enriched = await atlas_client.call_ability("enrich_records", {
        "customer": state['customer_name'],
        "entities": state['normalized_fields']
    })
    state['enriched_data'] = enriched
    
    # Add flags and calculations (COMMON)
    flags = await common_client.call_ability("add_flags_calculations", {
        "data": state['enriched_data']
    })
    state['flags_calculations'] = flags
    
    state['current_stage'] = 'PREPARE'
    state['stage_history'].append('PREPARE')
    
    logger.info(f"✅ Data normalized and enriched: Priority score {flags.get('priority_score')}")
    return state

async def stage_4_ask(state: WorkflowState) -> WorkflowState:
    """Stage 4: ASK - Request clarification if needed (Human interaction)"""
    logger.info("🔵 Stage 4: ASK (Human Interaction)")
    
    # Check if clarification needed
    if not state.get('email') or state['email'] == "":
        clarification = await atlas_client.call_ability("clarify_question", {
            "missing_fields": ["email"],
            "context": state['query']
        })
        state['clarification_needed'] = True
        state['clarification_question'] = clarification.get('question')
        logger.info(f"❓ Clarification needed: {state['clarification_question']}")
    else:
        state['clarification_needed'] = False
        logger.info("✅ No clarification needed")
    
    state['current_stage'] = 'ASK'
    state['stage_history'].append('ASK')
    return state

async def stage_5_wait(state: WorkflowState) -> WorkflowState:
    """Stage 5: WAIT - Extract and store answer (Deterministic)"""
    logger.info("🔵 Stage 5: WAIT (Deterministic)")
    
    if state.get('clarification_needed'):
        # Simulate customer response
        simulated_response = "john.smith@email.com"
        
        # Extract answer (ATLAS)
        answer = await atlas_client.call_ability("extract_answer", {
            "response": simulated_response
        })
        state['customer_response'] = answer.get('extracted')
        
        # Store answer in state
        if '@' in state['customer_response']:
            state['email'] = state['customer_response']
        
        logger.info(f"✅ Customer responded: {state['customer_response']}")
    else:
        logger.info("✅ No wait needed")
    
    state['current_stage'] = 'WAIT'
    state['stage_history'].append('WAIT')
    return state

async def stage_6_retrieve(state: WorkflowState) -> WorkflowState:
    """Stage 6: RETRIEVE - Search knowledge base (Deterministic)"""
    logger.info("🔵 Stage 6: RETRIEVE (Deterministic)")
    
    # Knowledge base search (ATLAS)
    kb_results = await atlas_client.call_ability("knowledge_base_search", {
        "query": state['query'],
        "intent": state['parsed_intent']
    })
    state['knowledge_base_results'] = kb_results.get('results', [])
    
    # Store relevant data (simulate database lookup)
    state['relevant_data'] = {
        "order_status": "shipped",
        "tracking_number": "1Z999AA1234567890",
        "estimated_delivery": "2025-08-29",
        "carrier": "UPS",
        "order_date": "2025-08-25",
        "shipping_address": "123 Main St, Anytown, USA"
    }
    
    state['current_stage'] = 'RETRIEVE'
    state['stage_history'].append('RETRIEVE')
    
    logger.info(f"✅ Retrieved {len(state['knowledge_base_results'])} KB articles")
    logger.info(f"✅ Relevant data: {state['relevant_data'].get('order_status', 'N/A')}")
    return state

async def stage_7_decide(state: WorkflowState) -> WorkflowState:
    """Stage 7: DECIDE - Generate and evaluate solutions (Non-deterministic)"""
    logger.info("🔵 Stage 7: DECIDE (Non-deterministic) - AI Solution Generation & Evaluation")
    
    # Enhanced solution generation and evaluation (COMMON)
    solution_results = await common_client.call_ability("solution_evaluation", {
        "query": state['query'],
        "intent": state['parsed_intent'],
        "kb_data": state['knowledge_base_results'],
        "relevant_data": state['relevant_data']
    })
    
    state['solutions'] = solution_results.get('solutions', [])
    
    # Choose best solution based on AI recommendation
    recommended_action = solution_results.get('recommended_solution')
    if recommended_action:
        best_solution = next((sol for sol in state['solutions'] if sol['action'] == recommended_action), None)
    else:
        best_solution = max(state['solutions'], key=lambda x: x.get('score', 0))
    
    state['chosen_solution'] = best_solution
    
    # Escalation decision (ATLAS)
    escalation = await atlas_client.call_ability("escalation_decision", {
        "score": best_solution.get('score', 0),
        "intent": state['parsed_intent']
    })
    state['escalation_required'] = escalation.get('escalate', False)
    
    # Override escalation based on AI solution recommendation
    if best_solution.get('requires_human', False):
        state['escalation_required'] = True
    
    state['current_stage'] = 'DECIDE'
    state['stage_history'].append('DECIDE')
    
    logger.info(f"✅ AI Generated {len(state['solutions'])} solutions")
    logger.info(f"✅ Chosen solution: {best_solution['action']} (score: {best_solution['score']})")
    logger.info(f"✅ Escalation required: {state['escalation_required']}")
    return state

async def stage_8_update(state: WorkflowState) -> WorkflowState:
    """Stage 8: UPDATE - Update ticket status (Deterministic)"""
    logger.info("🔵 Stage 8: UPDATE (Deterministic)")
    
    # Update ticket (ATLAS)
    ticket_update = await atlas_client.call_ability("update_ticket", {
        "ticket_id": state['ticket_id'],
        "status": "escalated" if state['escalation_required'] else "in_progress",
        "priority": state['flags_calculations'].get('priority_score', 50)
    })
    state['ticket_status'] = ticket_update.get('status')
    
    # Close ticket if resolved (ATLAS)
    if not state['escalation_required'] and state['chosen_solution']['score'] >= 90:
        close_result = await atlas_client.call_ability("close_ticket", {
            "ticket_id": state['ticket_id']
        })
        state['ticket_status'] = 'resolved'
    
    state['current_stage'] = 'UPDATE'
    state['stage_history'].append('UPDATE')
    
    logger.info(f"✅ Ticket status: {state['ticket_status']}")
    return state

async def stage_9_create(state: WorkflowState) -> WorkflowState:
    """Stage 9: CREATE - Generate response (Deterministic)"""
    logger.info("🔵 Stage 9: CREATE (Deterministic)")
    
    # Response generation (COMMON)
    response = await common_client.call_ability("response_generation", {
        "context": {
            "query": state['query'],
            "solution": state['chosen_solution'],
            "data": state['relevant_data'],
            "customer_name": state['customer_name']
        }
    })
    state['generated_response'] = response.get('response')
    
    state['current_stage'] = 'CREATE'
    state['stage_history'].append('CREATE')
    
    logger.info(f"✅ Response generated: {state['generated_response'][:100]}...")
    return state

async def stage_10_do(state: WorkflowState) -> WorkflowState:
    """Stage 10: DO - Execute actions (Deterministic)"""
    logger.info("🔵 Stage 10: DO (Deterministic)")
    
    # Execute API calls (ATLAS)
    api_results = await atlas_client.call_ability("execute_api_calls", {
        "ticket_id": state['ticket_id'],
        "actions": ["update_crm", "log_interaction"]
    })
    state['executed_actions'] = api_results.get('actions', [])
    
    # Trigger notifications (ATLAS)
    notifications = await atlas_client.call_ability("trigger_notifications", {
        "customer_email": state['email'],
        "message": state['generated_response']
    })
    state['notifications_sent'] = notifications.get('notifications', [])
    
    state['current_stage'] = 'DO'
    state['stage_history'].append('DO')
    
    logger.info(f"✅ Actions executed: {state['executed_actions']}")
    logger.info(f"✅ Notifications sent: {state['notifications_sent']}")
    return state

async def stage_11_complete(state: WorkflowState) -> WorkflowState:
    """Stage 11: COMPLETE - Output final payload"""
    logger.info("🔵 Stage 11: COMPLETE")
    
    # Calculate processing time
    start_time = datetime.fromisoformat(state['timestamp'])
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    state['processing_time'] = f"{processing_time:.2f} seconds"
    state['final_status'] = 'completed'
    state['current_stage'] = 'COMPLETE'
    state['stage_history'].append('COMPLETE')
    
    # Create final payload
    final_payload = {
        "ticket_id": state['ticket_id'],
        "status": state['ticket_status'],
        "customer": {
            "name": state['customer_name'],
            "email": state['email']
        },
        "analysis": {
            "intent": state['parsed_intent'],
            "sentiment": state['sentiment'],
            "urgency": state.get('urgency'),
            "complexity": state.get('complexity')
        },
        "solution": {
            "action": state['chosen_solution']['action'] if state.get('chosen_solution') else "N/A",
            "description": state['chosen_solution']['description'] if state.get('chosen_solution') else "N/A",
            "confidence_score": state['chosen_solution']['score'] if state.get('chosen_solution') else 0
        },
        "response_sent": state['generated_response'] is not None,
        "escalated": state['escalation_required'],
        "processing_time": state['processing_time'],
        "actions_taken": state['executed_actions'],
        "notifications_sent": state['notifications_sent'],
        "stages_completed": state['stage_history']
    }
    
    logger.info("✅ WORKFLOW COMPLETED")
    logger.info(f"📊 Final Payload:\n{json.dumps(final_payload, indent=2)}")
    
    return state
