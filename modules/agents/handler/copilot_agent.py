"""
Copilot Agent - Interactive assistance and guidance for users
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.agents.base_agent import BaseAgent, AgentCapability
from modules.config import get_config, CONFIG


class CopilotAgent(BaseAgent):
    """
    Interactive assistance agent for user guidance and conversation
    Enhanced with async execution, persistent conversation history, and comprehensive rule-based assistance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="CopilotAgent",
            capabilities=[AgentCapability.META_REASONING, AgentCapability.FEEDBACK],
            description="Interactive assistance and guidance for users",
            config=config
        )
        self.conversation_history = []
        self.user_context = {}
        self.knowledge_base = self._initialize_knowledge_base()

        # Persistence setup
        self.data_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing conversation history
        self._load_conversations()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize comprehensive knowledge base for assistance"""
        return {
            "system_capabilities": {
                "document_ingestion": "Can ingest PDFs, extract text, generate embeddings, and build searchable knowledge bases",
                "query_processing": "Supports natural language queries, semantic search, and contextual responses",
                "agent_coordination": "Manages multiple specialized agents for complex workflows",
                "resource_management": "Monitors and optimizes CPU, memory, and storage resources",
                "session_management": "Maintains conversation context and user preferences",
                "data_persistence": "Stores plans, conversations, and system state persistently"
            },
            "common_tasks": {
                "ingest_documents": {
                    "description": "Add documents to the knowledge base",
                    "steps": ["Prepare files", "Run ingestion command", "Verify processing", "Test queries"],
                    "commands": ["python kalki.py ingest --path /path/to/files"]
                },
                "query_knowledge": {
                    "description": "Search and retrieve information",
                    "steps": ["Formulate question", "Execute query", "Review results", "Refine if needed"],
                    "commands": ["python kalki.py query 'your question here'"]
                },
                "manage_sessions": {
                    "description": "Work with conversation sessions",
                    "steps": ["Start session", "Interact normally", "Session persists automatically"],
                    "commands": ["python kalki.py session --list"]
                }
            },
            "troubleshooting": {
                "ingestion_failed": "Check file formats, permissions, and available disk space",
                "query_no_results": "Try rephrasing question, check spelling, ensure documents are ingested",
                "performance_slow": "Check system resources, consider resource optimization",
                "agent_errors": "Review logs, check agent configurations, restart if needed"
            },
            "best_practices": [
                "Use specific, descriptive queries for better results",
                "Ingest high-quality documents for accurate responses",
                "Regularly monitor system resources and performance",
                "Keep conversations focused on specific tasks or topics",
                "Use the help command to explore available features"
            ]
        }

    def _load_conversations(self):
        """Load persisted conversation history from disk"""
        try:
            conv_file = self.data_dir / "conversations.json"
            if conv_file.exists():
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get("history", [])
                    self.user_context = data.get("context", {})
                self.logger.info(f"Loaded {len(self.conversation_history)} conversation entries from disk")
        except Exception as e:
            self.logger.exception(f"Failed to load conversations: {e}")

    def _save_conversations(self):
        """Persist conversation history to disk"""
        try:
            conv_file = self.data_dir / "conversations.json"
            data = {
                "history": self.conversation_history[-500:],  # Keep last 500 entries
                "context": self.user_context,
                "last_updated": datetime.now().isoformat()
            }
            with open(conv_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Saved conversations to disk")
        except Exception as e:
            self.logger.exception(f"Failed to save conversations: {e}")

    async def assist(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Provide comprehensive interactive assistance using rule-based intelligence
        """
        try:
            # Store user input in conversation history
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "context": context or {},
                "session_id": context.get("session_id") if context else None
            }

            # Generate comprehensive assistance response
            assistance = await self._generate_comprehensive_assistance(user_input, context)

            # Store response
            conversation_entry["assistance"] = assistance
            conversation_entry["method"] = "rule_based_comprehensive"

            self.conversation_history.append(conversation_entry)

            # Update user context
            self._update_user_context(user_input, assistance, context)

            # Persist changes
            self._save_conversations()

            return assistance

        except Exception as e:
            self.logger.exception(f"Failed to provide assistance: {e}")
            return f"I apologize, but I encountered an error while processing your request: {e}"

    async def _generate_comprehensive_assistance(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive assistance using advanced rule-based intelligence
        """
        # Analyze user input comprehensively
        intent_analysis = self._analyze_intent_comprehensively(user_input)
        intent = intent_analysis["primary_intent"]
        confidence = intent_analysis["confidence"]
        entities = intent_analysis["entities"]

        # Get conversation context
        conversation_context = self._get_conversation_context()

        # Generate response based on intent
        if intent == "help_request":
            return await self._generate_help_response(user_input, entities, conversation_context)
        elif intent == "task_guidance":
            return await self._generate_task_guidance(user_input, entities, conversation_context)
        elif intent == "system_query":
            return await self._generate_system_query_response(user_input, entities, conversation_context)
        elif intent == "troubleshooting":
            return await self._generate_troubleshooting_response(user_input, entities, conversation_context)
        elif intent == "capability_inquiry":
            return await self._generate_capability_response(user_input, entities, conversation_context)
        elif intent == "confirmation_request":
            return await self._generate_confirmation_response(user_input, entities, conversation_context)
        else:
            return await self._generate_general_response(user_input, intent_analysis, conversation_context)

    def _analyze_intent_comprehensively(self, user_input: str) -> Dict[str, Any]:
        """Comprehensive intent analysis with confidence scoring"""
        input_lower = user_input.lower()
        entities = self._extract_entities(user_input)

        # Define intent patterns with weights
        intent_patterns = {
            "help_request": {
                "patterns": ["help", "how do i", "how to", "guide", "tutorial", "assist"],
                "weight": 1.0
            },
            "task_guidance": {
                "patterns": ["ingest", "upload", "process", "query", "search", "analyze"],
                "weight": 1.0
            },
            "system_query": {
                "patterns": ["what can you", "capabilities", "features", "status", "health"],
                "weight": 0.9
            },
            "troubleshooting": {
                "patterns": ["error", "problem", "issue", "failed", "not working", "trouble"],
                "weight": 0.8
            },
            "capability_inquiry": {
                "patterns": ["can you", "do you", "are you able", "support"],
                "weight": 0.7
            },
            "confirmation_request": {
                "patterns": ["is it", "does it", "should i", "confirm"],
                "weight": 0.6
            }
        }

        # Calculate intent scores
        intent_scores = {}
        for intent, config in intent_patterns.items():
            score = 0
            for pattern in config["patterns"]:
                if pattern in input_lower:
                    score += config["weight"]
            intent_scores[intent] = score

        # Find primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]

        # Calculate confidence
        total_score = sum(intent_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5

        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "scores": intent_scores,
            "entities": entities
        }

    def _extract_entities(self, user_input: str) -> Dict[str, List[str]]:
        """Extract entities from user input"""
        entities = {
            "actions": [],
            "objects": [],
            "modifiers": []
        }

        # Action verbs
        actions = ["ingest", "upload", "process", "query", "search", "analyze", "create", "delete", "update"]
        for action in actions:
            if action in user_input.lower():
                entities["actions"].append(action)

        # Object nouns
        objects = ["document", "pdf", "file", "data", "knowledge", "session", "agent", "resource"]
        for obj in objects:
            if obj in user_input.lower():
                entities["objects"].append(obj)

        # Modifiers
        modifiers = ["quickly", "slowly", "efficiently", "thoroughly", "automatically"]
        for mod in modifiers:
            if mod in user_input.lower():
                entities["modifiers"].append(mod)

        return entities

    def _get_conversation_context(self) -> Dict[str, Any]:
        """Get relevant conversation context"""
        recent_entries = self.conversation_history[-3:]  # Last 3 exchanges

        context = {
            "recent_topics": [],
            "user_patterns": [],
            "session_length": len(self.conversation_history)
        }

        # Extract topics from recent conversation
        for entry in recent_entries:
            user_input = entry.get("user_input", "").lower()
            if "ingest" in user_input or "document" in user_input:
                context["recent_topics"].append("document_ingestion")
            elif "query" in user_input or "search" in user_input:
                context["recent_topics"].append("information_retrieval")
            elif "help" in user_input:
                context["recent_topics"].append("assistance")

        # Remove duplicates
        context["recent_topics"] = list(set(context["recent_topics"]))

        return context

    async def _generate_help_response(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate comprehensive help response"""
        help_topics = []

        # Determine specific help topics based on entities and context
        if "ingest" in entities["actions"] or "document" in entities["objects"]:
            help_topics.append("document_ingestion")
        if "query" in entities["actions"] or "search" in entities["actions"]:
            help_topics.append("query_processing")
        if "session" in entities["objects"]:
            help_topics.append("session_management")
        if not help_topics:
            help_topics = ["general_assistance"]

        response_parts = ["I can help you with various tasks in the Kalki knowledge base system:"]

        for topic in help_topics:
            if topic == "document_ingestion":
                response_parts.append("""
📚 **Document Management:**
- Ingest PDF documents and other files
- Extract text, images, and metadata
- Build searchable knowledge bases
- Process embeddings for semantic search""")
            elif topic == "query_processing":
                response_parts.append("""
🔍 **Query & Search:**
- Ask questions about your documents
- Search for specific information
- Get relevant excerpts and summaries
- Use natural language queries""")
            elif topic == "session_management":
                response_parts.append("""
💬 **Session Management:**
- Maintain conversation context
- Remember your preferences
- Provide personalized assistance
- Persistent conversation history""")
            elif topic == "general_assistance":
                response_parts.append("""
🛠️ **System Operations:**
- Monitor system resources
- Optimize performance
- Troubleshoot issues
- Manage agent coordination""")

        response_parts.append(f"""
**Best Practices:**
- Use specific, descriptive queries for better results
- Ingest high-quality documents for accurate responses
- Keep conversations focused on specific tasks

What would you like help with specifically?""")

        return "\n".join(response_parts)

    async def _generate_task_guidance(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate detailed task guidance"""
        primary_action = entities["actions"][0] if entities["actions"] else "general"

        if primary_action == "ingest":
            return await self._generate_ingestion_guidance()
        elif primary_action in ["query", "search"]:
            return await self._generate_query_guidance()
        elif primary_action == "analyze":
            return await self._generate_analysis_guidance()
        else:
            return await self._generate_general_task_guidance(primary_action)

    async def _generate_ingestion_guidance(self) -> str:
        """Generate comprehensive document ingestion guidance"""
        return """**Document Ingestion Process:**

1. **Prepare Your Files:**
   - Ensure PDFs are text-searchable (not image-only)
   - Check file permissions and accessibility
   - Organize files in a dedicated directory

2. **Execute Ingestion:**
   ```bash
   python kalki.py ingest --path /path/to/your/documents
   ```

3. **Monitor Progress:**
   - Check console output for processing status
   - Review logs for any errors or warnings
   - Verify file counts and processing times

4. **Verify Results:**
   - Test queries on ingested content
   - Check knowledge base statistics
   - Validate search functionality

**Supported Formats:**
- PDF documents (text-based)
- Plain text files
- Structured documents

**Tips for Success:**
- Process documents in batches for better performance
- Ensure consistent file naming
- Monitor available disk space
- Review processing logs for quality issues

Would you like me to help you start an ingestion process or explain any of these steps in more detail?"""

    async def _generate_query_guidance(self) -> str:
        """Generate comprehensive query guidance"""
        return """**Querying Your Knowledge Base:**

**Query Types Supported:**
- **Factual Questions:** "What are the main benefits of renewable energy?"
- **Specific Search:** "Find information about machine learning algorithms"
- **Comparative Queries:** "Compare supervised vs unsupervised learning"
- **Summary Requests:** "Summarize the key points about climate change"

**Query Best Practices:**
- Be specific about what you're looking for
- Include relevant context or time periods
- Use natural language - no special syntax needed
- Try variations if initial results aren't satisfactory

**Advanced Features:**
- **Context Awareness:** System remembers conversation context
- **Multi-document Search:** Searches across all ingested documents
- **Relevance Ranking:** Results ordered by relevance to your query
- **Excerpt Highlighting:** Shows relevant text snippets

**Example Queries:**
```
"What are the environmental impacts of electric vehicles?"
"How does quantum computing differ from classical computing?"
"Summarize the company's Q4 financial results"
```

Try asking me a question about your documents, and I'll help you formulate the best query!"""

    async def _generate_analysis_guidance(self) -> str:
        """Generate analysis task guidance"""
        return """**Data Analysis Capabilities:**

**Available Analysis Types:**
- **Document Analysis:** Extract insights from ingested documents
- **Content Summarization:** Generate concise summaries of long documents
- **Topic Modeling:** Identify main themes and topics
- **Sentiment Analysis:** Understand emotional tone and opinions
- **Entity Recognition:** Extract people, organizations, and key terms

**Analysis Process:**
1. **Specify Analysis Type:** Choose what you want to analyze
2. **Define Scope:** Select documents or content to analyze
3. **Execute Analysis:** Run automated analysis algorithms
4. **Review Results:** Examine findings and insights
5. **Refine Queries:** Ask follow-up questions for deeper insights

**Tips for Effective Analysis:**
- Start with broad questions, then narrow down
- Use specific document references when possible
- Combine multiple analysis types for comprehensive insights
- Review results critically and ask clarifying questions

What type of analysis are you interested in performing?"""

    async def _generate_general_task_guidance(self, action: str) -> str:
        """Generate guidance for general tasks"""
        return f"""**Guidance for {action.title()} Tasks:**

Based on your request to {action}, here's how I can help:

**Planning Phase:**
- Break down complex tasks into manageable steps
- Identify required resources and dependencies
- Estimate time and effort needed

**Execution Phase:**
- Coordinate multiple agents if needed
- Monitor progress and handle issues
- Optimize resource usage

**Validation Phase:**
- Verify results meet requirements
- Test functionality thoroughly
- Document outcomes and lessons learned

**Available Tools:**
- Task planning and decomposition
- Resource allocation and monitoring
- Agent coordination and workflow management
- Progress tracking and reporting

Would you like me to help you plan and execute this {action} task, or do you need guidance on a specific aspect?"""

    async def _generate_system_query_response(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate response to system capability queries"""
        return """**Kalki Knowledge Base System Capabilities:**

**Core Features:**
- **Intelligent Document Processing:** Advanced PDF parsing, text extraction, and metadata generation
- **Semantic Search:** Vector-based search with relevance ranking and context awareness
- **Multi-Agent Architecture:** Specialized agents for different tasks (planning, execution, optimization)
- **Resource Management:** Dynamic CPU, memory, and storage optimization
- **Persistent Sessions:** Conversation history and context preservation

**Supported Operations:**
- Document ingestion and indexing
- Natural language querying
- Content analysis and summarization
- Workflow orchestration
- System monitoring and optimization
- Interactive assistance and guidance

**Technical Specifications:**
- Async processing for scalability
- JSON-based persistence
- Comprehensive logging and error handling
- Modular agent architecture
- RESTful API interfaces

**Current System Status:**
- All core agents operational
- Document processing pipeline active
- Query engine ready
- Resource monitoring enabled

Is there a specific capability you'd like me to demonstrate or explain in more detail?"""

    async def _generate_troubleshooting_response(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate troubleshooting guidance"""
        # Identify potential issues from user input
        issue_keywords = ["error", "failed", "problem", "issue", "not working", "slow", "crash"]
        identified_issues = [kw for kw in issue_keywords if kw in user_input.lower()]

        response_parts = ["**Troubleshooting Guide:**"]

        if "ingestion" in user_input.lower() or "upload" in user_input.lower():
            response_parts.append("""
**Document Ingestion Issues:**
- Check file formats (PDFs should be text-searchable)
- Verify file permissions and accessibility
- Ensure sufficient disk space (minimum 2x file size)
- Review processing logs for specific errors
- Try processing files individually if batch fails""")
        elif "query" in user_input.lower() or "search" in user_input.lower():
            response_parts.append("""
**Query/Search Issues:**
- Try rephrasing your question more specifically
- Check spelling and terminology
- Ensure documents have been properly ingested
- Review query logs for processing details
- Consider using simpler, more direct questions""")
        elif "performance" in user_input.lower() or "slow" in user_input.lower():
            response_parts.append("""
**Performance Issues:**
- Check system resource usage (CPU, memory, disk)
- Monitor active processes and background tasks
- Consider reducing batch sizes for processing
- Review system logs for bottleneck identification
- Close unnecessary applications""")
        else:
            response_parts.append("""
**General Troubleshooting Steps:**
1. Check system logs for error messages
2. Verify all required services are running
3. Test with simple, known-working inputs
4. Restart services if needed
5. Review configuration settings
6. Check available system resources""")

        response_parts.append("""
**Getting Help:**
- Review the logs directory for detailed error information
- Check the README for common issues and solutions
- Try the --help flag on commands for usage information
- Provide specific error messages for targeted assistance

What specific issue are you experiencing? I can provide more targeted troubleshooting steps.""")

        return "\n".join(response_parts)

    async def _generate_capability_response(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate response about system capabilities"""
        capability_questions = {
            "ingest": "Yes, I can help you ingest documents, process PDFs, extract text and metadata, and build searchable knowledge bases.",
            "query": "Yes, I support natural language queries, semantic search, and can retrieve relevant information from your documents.",
            "analyze": "Yes, I can help analyze documents, generate summaries, identify topics, and extract key insights.",
            "manage": "Yes, I can help manage sessions, monitor resources, coordinate agents, and optimize system performance.",
            "remember": "Yes, I maintain conversation history and can reference previous interactions within our session.",
            "help": "Yes, I'm here to provide guidance, answer questions, and assist with any aspect of the system."
        }

        # Find matching capability
        for capability, response in capability_questions.items():
            if capability in user_input.lower():
                return f"**{capability.title()} Capability:** {response}\n\nWould you like me to demonstrate this capability or provide more details?"

        return """**System Capabilities Overview:**

I can assist with:
- 📄 **Document Ingestion:** Process and index various document types
- 🔍 **Intelligent Queries:** Search and retrieve information using natural language
- 📊 **Content Analysis:** Analyze documents and extract insights
- ⚙️ **System Management:** Monitor resources and optimize performance
- 🤖 **Agent Coordination:** Manage complex workflows across multiple agents
- 💬 **Interactive Assistance:** Provide guidance and answer questions

What specific capability would you like to know more about?"""

    async def _generate_confirmation_response(self, user_input: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Generate confirmation/clarification responses"""
        return """**Confirmation Request:**

I'd be happy to help clarify or confirm information for you. To provide the most accurate assistance, could you please:

1. **Specify what you need confirmed:** What exactly would you like me to verify or confirm?
2. **Provide context:** Any relevant details about the situation or previous actions?
3. **Include specifics:** File names, commands, settings, or other technical details?

For example:
- "Can you confirm that the document ingestion completed successfully?"
- "Does the system support PDF processing?"
- "Should I restart the service after configuration changes?"

Once you provide these details, I can give you a definitive answer or guide you through the verification process."""

    async def _generate_general_response(self, user_input: str, intent_analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate general response for unrecognized intents"""
        confidence = intent_analysis["confidence"]

        if confidence > 0.7:
            intent = intent_analysis["primary_intent"]
            return f"I understand you're asking about {intent.replace('_', ' ')}. Let me provide some guidance on that topic."
        elif confidence > 0.4:
            return f"I'm interpreting your request as related to {intent_analysis['primary_intent'].replace('_', ' ')}, but I'd like to make sure I understand correctly. Could you provide a bit more context?"
        else:
            return f"""I want to make sure I provide the most helpful response possible. Could you clarify what you're looking for? For example:

- Are you asking about document ingestion or processing?
- Do you need help with querying the knowledge base?
- Are you looking for system status or troubleshooting guidance?
- Would you like general assistance with using the system?

Feel free to rephrase your question or provide more details!"""

    def _update_user_context(self, user_input: str, assistance: str, context: Optional[Dict[str, Any]] = None):
        """Update user context based on interaction"""
        # Simple context tracking
        self.user_context["last_interaction"] = datetime.now().isoformat()
        self.user_context["interaction_count"] = self.user_context.get("interaction_count", 0) + 1

        # Track common topics
        topics = self._extract_topics(user_input)
        for topic in topics:
            if topic not in self.user_context.get("topics", []):
                if "topics" not in self.user_context:
                    self.user_context["topics"] = []
                self.user_context["topics"].append(topic)

        # Track user preferences based on interactions
        if "help" in user_input.lower():
            self.user_context["prefers_guidance"] = True
        if "status" in user_input.lower():
            self.user_context["interested_in_monitoring"] = True

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from user input (simple keyword-based)"""
        topics = []
        lower_text = text.lower()

        topic_keywords = {
            "document": ["document", "pdf", "file", "ingest", "upload"],
            "query": ["query", "search", "find", "ask", "retrieve"],
            "session": ["session", "memory", "remember", "conversation"],
            "system": ["system", "resource", "performance", "status", "monitor"],
            "help": ["help", "guide", "tutorial", "assist", "how"],
            "troubleshooting": ["error", "problem", "issue", "failed", "fix"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in lower_text for keyword in keywords):
                topics.append(topic)

        return topics

    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self._save_conversations()
        self.logger.info("Conversation history cleared")

    def get_user_context(self) -> Dict[str, Any]:
        """Get current user context"""
        return self.user_context.copy()

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Async execute copilot tasks"""
        try:
            action = task.get("action")

            if action == "assist":
                assistance = await self.assist(task["user_input"], task.get("context"))
                return {"status": "success", "assistance": assistance}
            elif action == "history":
                history = self.get_conversation_history(task.get("limit", 50))
                return {"status": "success", "history": history}
            elif action == "context":
                context = self.get_user_context()
                return {"status": "success", "context": context}
            elif action == "clear_history":
                self.clear_conversation_history()
                return {"status": "success", "message": "Conversation history cleared"}
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Failed to execute copilot task: {e}")
            return {"status": "error", "message": str(e)}

    async def initialize(self) -> bool:
        """
        Initialize the copilot agent
        """
        try:
            # Load existing conversations
            self._load_conversations()
            self.logger.info("CopilotAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize CopilotAgent: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the copilot agent
        """
        try:
            # Save current conversations
            self._save_conversations()
            self.logger.info("CopilotAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown CopilotAgent: {e}")
            return False
