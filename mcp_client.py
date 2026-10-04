import os
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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
                "status": "escalated" if params.get('status') == 'escalated' else "in_progress",
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

# Pre-initialized client instances
common_client = MCPClient(MCPServer.COMMON)
atlas_client = MCPClient(MCPServer.ATLAS)
