import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime
import google.generativeai as genai
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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

# ============= MCP Client Simulators =============
class MCPServer(Enum):
    COMMON = "COMMON"  # Internal AI processing
    ATLAS = "ATLAS"    # External system integrations

class MCPClient:
    """Simulates MCP Client for ability execution"""
    
    def __init__(self, server_type: MCPServer):
        self.server_type = server_type
        self.logger = logging.getLogger(f"MCP_{server_type.value}")
    
    async def call_ability(self, ability_name: str, params: Dict) -> Dict:
        self.logger.info(f"Calling {ability_name} on {self.server_type.value} with params: {params}")
        
        if self.server_type == MCPServer.COMMON:
            return await self._execute_common_ability(ability_name, params)
        else:
            return await self._execute_atlas_ability(ability_name, params)
    
    async def _execute_common_ability(self, ability_name: str, params: Dict) -> Dict:
        """Execute COMMON server abilities (AI processing)"""
        
        if ability_name == "parse_request_text":
            # Enhanced AI-powered text parsing using Gemini
            query_text = params.get('text', '')
            
            prompt = f"""
            As a customer service AI, analyze this customer query and extract key information:

            Customer Query: "{query_text}"

            Please provide a detailed analysis in JSON format with these fields:
            1. "intent": Categorize as one of [order_status, refund, technical_support, billing, general_inquiry, complaint, compliment, product_info]
            2. "sentiment": Analyze emotional tone [positive, neutral, negative, frustrated, satisfied]
            3. "urgency": Assess priority level [low, medium, high, critical]
            4. "topic_keywords": List 3-5 main topics/keywords from the query
            5. "customer_emotion": Detect emotional state [calm, anxious, angry, happy, confused]
            6. "complexity": Rate complexity [simple, moderate, complex]
            7. "requires_escalation": Boolean - does this need human intervention?

            Return only valid JSON format.
            """
            
            try:
                response = gemini_model.generate_content(prompt)
                # Clean up the response to extract JSON
                response_text = response.text.strip()
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    response_text = response_text.split('```')[1]
                
                result = json.loads(response_text)
                
                # Validate required fields
                required_fields = ['intent', 'sentiment', 'urgency']
                for field in required_fields:
                    if field not in result:
                        result[field] = 'unknown'
                        
            except Exception as e:
                logger.warning(f"AI parsing failed, using fallback: {e}")
                # Fallback logic
                result = {
                    "intent": "general_inquiry",
                    "sentiment": "neutral", 
                    "urgency": "medium",
                    "topic_keywords": ["support", "help"],
                    "customer_emotion": "calm",
                    "complexity": "moderate",
                    "requires_escalation": False
                }
            
            return result
        
        elif ability_name == "normalize_fields":
            # Normalize data fields
            data = params.get('data', {})
            normalized = {}
            for key, value in data.items():
                if key == "order_id" and isinstance(value, str):
                    normalized[key] = value.replace('#', '').strip()
                elif key == "last_name" and isinstance(value, str):
                    normalized[key] = value.upper()
                else:
                    normalized[key] = value
            return {"normalized": normalized}
        
        elif ability_name == "add_flags_calculations":
            # Calculate priority scores
            return {
                "priority_score": 75,
                "sla_risk": "low",
                "estimated_resolution_time": "2 hours"
            }
        
        elif ability_name == "solution_evaluation":
            # Enhanced AI-powered solution generation and evaluation
            query = params.get('query', '')
            intent = params.get('intent', 'general')
            kb_data = params.get('kb_data', [])
            relevant_data = params.get('relevant_data', {})
            
            # Prepare knowledge base context
            kb_context = ""
            if kb_data:
                kb_context = "\n".join([f"- {item.get('title', '')}: {item.get('content', '')}" for item in kb_data])
            
            prompt = f"""
            As a customer service AI, analyze this support query and generate solutions:

            Customer Query: "{query}"
            Intent: {intent}
            Customer Data: {json.dumps(relevant_data, indent=2)}
            
            Knowledge Base Information:
            {kb_context}

            Task: Generate 3 different solution approaches and evaluate each one.

            For each solution, provide:
            1. "action": Brief action name (e.g., "provide_tracking", "process_refund")
            2. "description": Detailed explanation of what to do
            3. "score": Confidence score 1-100 (higher = better solution)
            4. "reasoning": Why this solution fits
            5. "estimated_time": How long this solution takes
            6. "requires_human": Boolean - needs human agent involvement

            Return JSON format:
            {{
              "solutions": [
                {{
                  "action": "action_name",
                  "description": "detailed description",
                  "score": 95,
                  "reasoning": "why this works",
                  "estimated_time": "5 minutes",
                  "requires_human": false
                }}
              ],
              "recommended_solution": "action_name of best solution",
              "confidence_level": "high/medium/low",
              "escalation_reason": "reason if escalation needed"
            }}
            """
            
            try:
                response = gemini_model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Clean up JSON response
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    response_text = response_text.split('```')[1]
                
                result = json.loads(response_text)
                
                # Validate structure
                if 'solutions' not in result or not result['solutions']:
                    raise ValueError("No solutions generated")
                
                # Ensure all solutions have required fields
                for solution in result['solutions']:
                    if 'score' not in solution:
                        solution['score'] = 50
                    if 'action' not in solution:
                        solution['action'] = 'general_support'
                    if 'description' not in solution:
                        solution['description'] = 'Provide general assistance'
                
            except Exception as e:
                logger.warning(f"AI solution generation failed, using fallback: {e}")
                # Fallback solutions based on intent
                fallback_solutions = {
                    'order_status': [
                        {"action": "provide_tracking", "description": "Provide order tracking information", "score": 90, "reasoning": "Direct tracking info resolves query", "estimated_time": "2 minutes", "requires_human": False},
                        {"action": "check_warehouse", "description": "Verify order status with warehouse", "score": 75, "reasoning": "More detailed status check", "estimated_time": "10 minutes", "requires_human": False},
                        {"action": "escalate_shipping", "description": "Escalate to shipping specialist", "score": 40, "reasoning": "Complex shipping issue", "estimated_time": "30 minutes", "requires_human": True}
                    ],
                    'refund': [
                        {"action": "process_refund", "description": "Process automatic refund", "score": 85, "reasoning": "Meets refund criteria", "estimated_time": "5 minutes", "requires_human": False},
                        {"action": "review_refund", "description": "Manual refund review required", "score": 60, "reasoning": "Complex case needs review", "estimated_time": "24 hours", "requires_human": True}
                    ],
                    'default': [
                        {"action": "provide_info", "description": "Provide relevant information", "score": 70, "reasoning": "General support response", "estimated_time": "5 minutes", "requires_human": False},
                        {"action": "escalate_general", "description": "Transfer to human agent", "score": 50, "reasoning": "Complex query needs human touch", "estimated_time": "15 minutes", "requires_human": True}
                    ]
                }
                
                solutions = fallback_solutions.get(intent, fallback_solutions['default'])
                result = {
                    "solutions": solutions,
                    "recommended_solution": solutions[0]['action'],
                    "confidence_level": "medium",
                    "escalation_reason": ""
                }
            
            return result
        
        elif ability_name == "response_generation":
            # Generate customer response using Gemini
            context = params.get('context', {})
            prompt = f"""Generate a professional customer support response for:
            Query: {context.get('query')}
            Solution: {context.get('solution')}
            Data: {context.get('data')}
    
            Be friendly, concise, and helpful."""
    
            try:
                response = gemini_model.generate_content(prompt)
                return {"response": response.text}
            except Exception as e:
                logger.warning(f"AI response generation failed, using fallback: {e}")
                # Fallback response
                return {"response": f"Thank you for contacting us, {context.get('customer_name', 'valued customer')}. We're looking into your request and will get back to you soon."}
        
        return {"status": "completed", "ability": ability_name}
    
    async def _execute_atlas_ability(self, ability_name: str, params: Dict) -> Dict:
        """Execute ATLAS server abilities (external integrations)"""
        
        if ability_name == "extract_entities":
            # Extract entities from text
            text = params.get('text', '')
            entities = {}
            
            # Simple extraction logic (in production, use NER or regex)
            if '#' in text:
                order_start = text.find('#')
                order_end = text.find(' ', order_start)
                if order_end == -1:
                    order_end = len(text)
                entities['order_id'] = text[order_start:order_end]
            
            if 'last name is' in text.lower():
                name_start = text.lower().find('last name is') + 12
                name_end = text.find(' ', name_start)
                if name_end == -1:
                    name_end = len(text)
                entities['last_name'] = text[name_start:name_end].strip()
            
            return {"entities": entities}
        
        elif ability_name == "enrich_records":
            # Simulate database lookup
            return {
                "customer_tier": "premium",
                "previous_complaints": 0,
                "sla_deadline": "2025-08-30",
                "account_status": "active"
            }
        
        elif ability_name == "clarify_question":
            clarification = input("Clarification needed. Please provide additional info: ")
            return {
                "question": "Could you provide your email address for verification?",
                "required_fields": []
            }
            return {
                "question": "Could you provide your email address for verification?",
                "required_fields": ["email"]
            }
        
        elif ability_name == "extract_answer":
            # Extract answer from customer response
            return {"extracted": params.get('response', '')}
        
        elif ability_name == "knowledge_base_search":
            # Simulate KB search
            query = params.get('query', '')
            results = [
                {
                    "title": "Order Tracking Guide",
                    "content": "Track your order using the tracking number provided in your confirmation email. Orders typically ship within 1-2 business days.",
                    "relevance": 0.95
                },
                {
                    "title": "Delivery Timeline",
                    "content": "Standard delivery takes 3-5 business days. Express delivery is available for 1-2 business days.",
                    "relevance": 0.82
                },
                {
                    "title": "Order Status Meanings",
                    "content": "Processing: Order received, Shipped: On the way, Delivered: Package received",
                    "relevance": 0.78
                }
            ]
            return {"results": results}
        
        elif ability_name == "escalation_decision":
            # Decide on escalation
            score = params.get('score', 100)
            return {"escalate": score < 90, "reason": "Score below threshold" if score < 90 else ""}
        
        elif ability_name == "update_ticket":
            # Update ticket in system
            return {
                "ticket_id": params.get('ticket_id'),
                "status": "in_progress",
                "updated_at": datetime.now().isoformat()
            }
        
        elif ability_name == "close_ticket":
            # Close ticket
            return {
                "ticket_id": params.get('ticket_id'),
                "status": "resolved",
                "closed_at": datetime.now().isoformat()
            }
        
        elif ability_name == "execute_api_calls":
            # Execute external API calls
            return {
                "crm_updated": True,
                "order_system_notified": True,
                "actions": ["crm_update", "order_notification"]
            }
        
        elif ability_name == "trigger_notifications":
            # Send notifications
            return {
                "email_sent": True,
                "sms_sent": False,
                "notifications": ["email_confirmation"]
            }
        
        return {"status": "completed", "ability": ability_name}

# ============= Stage Implementations =============

# Initialize MCP Clients
common_client = MCPClient(MCPServer.COMMON)
atlas_client = MCPClient(MCPServer.ATLAS)

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

# ============= LangGraph Workflow Builder =============
def build_customer_support_workflow():
    """Build the LangGraph workflow with 11 stages"""
    
    # Create workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes (stages)
    workflow.add_node("intake", stage_1_intake)
    workflow.add_node("understand", stage_2_understand)
    workflow.add_node("prepare", stage_3_prepare)
    workflow.add_node("ask", stage_4_ask)
    workflow.add_node("wait", stage_5_wait)
    workflow.add_node("retrieve", stage_6_retrieve)
    workflow.add_node("decide", stage_7_decide)
    workflow.add_node("update", stage_8_update)
    workflow.add_node("create", stage_9_create)
    workflow.add_node("do", stage_10_do)
    workflow.add_node("complete", stage_11_complete)
    
    # Define edges (flow between stages)
    workflow.add_edge("intake", "understand")
    workflow.add_edge("understand", "prepare")
    workflow.add_edge("prepare", "ask")
    workflow.add_edge("ask", "wait")
    workflow.add_edge("wait", "retrieve")
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("decide", "update")
    workflow.add_edge("update", "create")
    workflow.add_edge("create", "do")
    workflow.add_edge("do", "complete")
    workflow.add_edge("complete", END)
    
    workflow.set_entry_point("intake")
    
    return workflow.compile()

# ============= Main Execution =============
async def run_customer_support_agent(input_payload: Dict):
    print("\n" + "="*60)
    print("LANGIE - Customer Support Agent Starting")
    print("="*60 + "\n")
    
    # Initialize state
    initial_state = WorkflowState(
        customer_name=input_payload.get('customer_name', ''),
        email=input_payload.get('email', ''),
        query=input_payload.get('query', ''),
        priority=input_payload.get('priority', ''),
        ticket_id=input_payload.get('ticket_id', ''),
        parsed_intent=None,
        entities=None,
        sentiment=None,
        normalized_fields=None,
        enriched_data=None,
        flags_calculations=None,
        clarification_needed=None,
        clarification_question=None,
        customer_response=None,
        knowledge_base_results=None,
        relevant_data=None,
        solutions=None,
        chosen_solution=None,
        escalation_required=None,
        ticket_status=None,
        generated_response=None,
        executed_actions=None,
        notifications_sent=None,
        final_status=None,
        processing_time=None,
        stage_history=[],
        current_stage='',
        timestamp=''
    )
    
    # Build workflow
    workflow = build_customer_support_workflow()
    
    # Execute workflow
    final_state = await workflow.ainvoke(initial_state)
    
    print("\n" + "="*60)
    print("✅ WORKFLOW EXECUTION COMPLETE")
    print("="*60 + "\n")
    
    return final_state

async def demo():
    sample_input = {
        "customer_name": "John Smith",
        "email": "",
        "query": "When will my order come? My order id is #456 and my last name is Smith",
        "priority": "medium",
        "ticket_id": "T001"
    }
    
    print("📝 Input Payload:")
    print(json.dumps(sample_input, indent=2))
    print("\n" + "="*60 + "\n")
    
    result = await run_customer_support_agent(sample_input)

    print()

if __name__ == "__main__":
    asyncio.run(demo())