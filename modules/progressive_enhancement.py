"""
Progressive Enhancement Engine
Intelligent query routing with confidence-based escalation for optimal speed/accuracy tradeoff.

Query Processing Levels:
Level 1: Fast text model (3.1 8B) - 2-3s response time
Level 2: Text + RAG retrieval - 3-4s response time  
Level 3: Vision model analysis (3.2 11B) - 5-8s response time
Level 4: Multi-agent consensus - 10-15s response time

Expected Distribution:
- 80% of queries: Level 1-2 (fast, text-only)
- 19% of queries: Level 3 (requires vision)
- 1% of queries: Level 4 (critical/complex)
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import time
import logging


logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Direct factual answer
    MODERATE = "moderate"       # Requires retrieval
    COMPLEX = "complex"         # Requires reasoning
    VISUAL = "visual"           # Requires image analysis
    CRITICAL = "critical"       # Requires high confidence


class ProcessingLevel(Enum):
    """Progressive processing levels"""
    LEVEL_1_FAST_TEXT = 1      # Fast text model only
    LEVEL_2_TEXT_RAG = 2       # Text + retrieval
    LEVEL_3_VISION = 3         # Vision model
    LEVEL_4_CONSENSUS = 4      # Multi-agent validation


@dataclass
class QueryAnalysis:
    """Analysis of incoming query"""
    query: str
    complexity: QueryComplexity
    requires_vision: bool
    requires_retrieval: bool
    requires_consensus: bool
    confidence_threshold: float
    estimated_processing_time: float


@dataclass
class QueryResult:
    """Result from query processing"""
    answer: str
    processing_level: ProcessingLevel
    confidence: float
    processing_time: float
    sources: List[str]
    reasoning: Optional[str]
    escalated_from: Optional[ProcessingLevel]
    metadata: Dict[str, Any]


class ProgressiveEnhancementEngine:
    """
    Intelligent query router with confidence-based escalation.
    
    Optimizes for speed while maintaining accuracy through progressive
    enhancement - starts fast, escalates only when needed.
    """
    
    def __init__(
        self,
        llm_engine,
        vision_engine=None,
        rag_system=None,
        consensus_system=None,
        cache=None
    ):
        """
        Initialize progressive enhancement engine.
        
        Args:
            llm_engine: Text LLM (Llama 3.1 8B)
            vision_engine: Vision LLM (Llama 3.2 11B)
            rag_system: RAG retrieval system
            consensus_system: Multi-agent consensus system
            cache: Vision cache for performance
        """
        self.llm = llm_engine
        self.vision = vision_engine
        self.rag = rag_system
        self.consensus = consensus_system
        self.cache = cache
        
        # Confidence thresholds for escalation
        self.confidence_thresholds = {
            QueryComplexity.SIMPLE: 0.85,
            QueryComplexity.MODERATE: 0.80,
            QueryComplexity.COMPLEX: 0.75,
            QueryComplexity.VISUAL: 0.70,
            QueryComplexity.CRITICAL: 0.95
        }
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "level_1_count": 0,
            "level_2_count": 0,
            "level_3_count": 0,
            "level_4_count": 0,
            "escalations": 0,
            "avg_response_time": 0.0
        }
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_level: Optional[ProcessingLevel] = None
    ) -> QueryResult:
        """
        Process query with progressive enhancement.
        
        Args:
            query: User's question
            context: Optional context (domain, images, etc.)
            force_level: Optional forced processing level (for testing)
        
        Returns:
            QueryResult with answer and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Analyze query to determine strategy
        analysis = self._analyze_query(query, context)
        
        logger.info(f"Query complexity: {analysis.complexity.value}")
        logger.info(f"Requires vision: {analysis.requires_vision}")
        logger.info(f"Requires retrieval: {analysis.requires_retrieval}")
        
        # Determine starting level
        if force_level:
            current_level = force_level
        else:
            current_level = self._determine_starting_level(analysis)
        
        logger.info(f"Starting at Level {current_level.value}")
        
        # Try processing at current level
        result = None
        escalated_from = None
        
        while result is None:
            try:
                if current_level == ProcessingLevel.LEVEL_1_FAST_TEXT:
                    result = await self._process_level_1(query, analysis)
                    self.stats["level_1_count"] += 1
                
                elif current_level == ProcessingLevel.LEVEL_2_TEXT_RAG:
                    result = await self._process_level_2(query, analysis, context)
                    self.stats["level_2_count"] += 1
                
                elif current_level == ProcessingLevel.LEVEL_3_VISION:
                    result = await self._process_level_3(query, analysis, context)
                    self.stats["level_3_count"] += 1
                
                elif current_level == ProcessingLevel.LEVEL_4_CONSENSUS:
                    result = await self._process_level_4(query, analysis, context)
                    self.stats["level_4_count"] += 1
                
                # Check if confidence is sufficient
                if result and not self._is_confidence_sufficient(result, analysis):
                    # Escalate to next level
                    logger.info(
                        f"Confidence {result.confidence:.2f} below threshold "
                        f"{analysis.confidence_threshold:.2f} - escalating"
                    )
                    escalated_from = current_level
                    current_level = self._escalate(current_level)
                    self.stats["escalations"] += 1
                    result = None  # Retry at higher level
                
            except Exception as e:
                logger.error(f"Error at level {current_level.value}: {e}")
                # Try to escalate on error
                if current_level.value < 4:
                    escalated_from = current_level
                    current_level = self._escalate(current_level)
                    result = None
                else:
                    # Already at highest level, return error
                    result = QueryResult(
                        answer=f"Error processing query: {e}",
                        processing_level=current_level,
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        sources=[],
                        reasoning=None,
                        escalated_from=escalated_from,
                        metadata={"error": str(e)}
                    )
        
        # Update result with escalation info
        result.escalated_from = escalated_from
        result.processing_time = time.time() - start_time
        
        # Update statistics
        self._update_stats(result)
        
        logger.info(
            f"Query completed at Level {result.processing_level.value} "
            f"in {result.processing_time:.2f}s (confidence: {result.confidence:.2f})"
        )
        
        return result
    
    # Level processors
    
    async def _process_level_1(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> QueryResult:
        """
        Level 1: Fast text model (2-3s)
        
        Use for: Simple factual questions, definitions, calculations
        """
        logger.info("Level 1: Fast text processing")
        
        # Use fast text model without retrieval
        prompt = f"""Answer this question directly and concisely:

Question: {query}

Provide a clear, accurate answer. If you're not confident, say "I need to check my knowledge base for this."`"""
        
        response = await self.llm.generate(prompt)
        
        # Extract answer and confidence
        answer = response.get("text", "")
        confidence = self._estimate_confidence(answer, query)
        
        # Check for uncertainty markers
        uncertainty_phrases = [
            "i need to check",
            "i'm not sure",
            "i don't know",
            "may not be accurate",
            "need more information"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence = 0.3  # Force escalation
        
        return QueryResult(
            answer=answer,
            processing_level=ProcessingLevel.LEVEL_1_FAST_TEXT,
            confidence=confidence,
            processing_time=0.0,  # Will be updated
            sources=["llm_direct"],
            reasoning="Fast text generation without retrieval",
            escalated_from=None,
            metadata={"model": "llama_3.1_8b"}
        )
    
    async def _process_level_2(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> QueryResult:
        """
        Level 2: Text + RAG (3-4s)
        
        Use for: Domain-specific questions requiring retrieved knowledge
        """
        logger.info("Level 2: Text + RAG retrieval")
        
        # Retrieve relevant context
        retrieved_docs = []
        if self.rag and analysis.requires_retrieval:
            retrieved_docs = await self.rag.retrieve(query, top_k=5)
        
        # Build enriched prompt
        context_text = "\n\n".join([
            f"[Source {i+1}]: {doc['content']}"
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        prompt = f"""Answer this question using the provided context:

Context:
{context_text}

Question: {query}

Provide a detailed answer based on the context. Cite sources when possible.
If the context doesn't contain enough information, say so."""
        
        response = await self.llm.generate(prompt)
        
        answer = response.get("text", "")
        confidence = self._estimate_confidence_with_retrieval(
            answer, 
            query, 
            retrieved_docs
        )
        
        sources = [doc.get("source", "unknown") for doc in retrieved_docs]
        
        return QueryResult(
            answer=answer,
            processing_level=ProcessingLevel.LEVEL_2_TEXT_RAG,
            confidence=confidence,
            processing_time=0.0,
            sources=sources,
            reasoning="Answer generated from retrieved knowledge",
            escalated_from=None,
            metadata={
                "retrieved_docs": len(retrieved_docs),
                "model": "llama_3.1_8b"
            }
        )
    
    async def _process_level_3(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> QueryResult:
        """
        Level 3: Vision model (5-8s)
        
        Use for: Queries requiring image analysis
        """
        logger.info("Level 3: Vision model processing")
        
        if not self.vision:
            raise Exception("Vision engine not available")
        
        # Extract images from context
        images = context.get("images", []) if context else []
        
        if not images:
            raise Exception("No images provided for visual analysis")
        
        # Analyze images with vision model
        vision_results = []
        for image_path in images:
            result = await self._analyze_image_cached(image_path, query)
            vision_results.append(result)
        
        # Combine vision results with text reasoning
        vision_text = "\n\n".join([
            f"Image {i+1} Analysis:\n{result.get('analysis', '')}"
            for i, result in enumerate(vision_results)
        ])
        
        # Generate final answer combining vision and text
        prompt = f"""Answer this question using the visual analysis:

Question: {query}

Visual Analysis:
{vision_text}

Provide a comprehensive answer that incorporates the visual information."""
        
        response = await self.llm.generate(prompt)
        
        answer = response.get("text", "")
        confidence = min(
            self._estimate_confidence(answer, query),
            *[r.get("confidence", 0.85) for r in vision_results]
        )
        
        return QueryResult(
            answer=answer,
            processing_level=ProcessingLevel.LEVEL_3_VISION,
            confidence=confidence,
            processing_time=0.0,
            sources=[f"vision_analysis_{i}" for i in range(len(images))],
            reasoning="Answer generated from visual analysis",
            escalated_from=None,
            metadata={
                "images_analyzed": len(images),
                "vision_model": "llama_3.2_11b_vision"
            }
        )
    
    async def _process_level_4(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> QueryResult:
        """
        Level 4: Multi-agent consensus (10-15s)
        
        Use for: Critical decisions requiring high confidence
        """
        logger.info("Level 4: Multi-agent consensus")
        
        if not self.consensus:
            # Fallback: just use highest confidence from previous levels
            logger.warning("Consensus system not available, using enhanced single model")
            
            # Try both text and vision approaches
            text_result = await self._process_level_2(query, analysis, context)
            
            vision_result = None
            if self.vision and context and context.get("images"):
                try:
                    vision_result = await self._process_level_3(query, analysis, context)
                except:
                    pass
            
            # Return result with higher confidence
            if vision_result and vision_result.confidence > text_result.confidence:
                return vision_result
            else:
                return text_result
        
        # Use consensus system for multi-agent validation
        consensus_result = await self.consensus.get_consensus(
            query,
            context,
            require_unanimous=True
        )
        
        return QueryResult(
            answer=consensus_result.get("answer", ""),
            processing_level=ProcessingLevel.LEVEL_4_CONSENSUS,
            confidence=consensus_result.get("confidence", 0.95),
            processing_time=0.0,
            sources=consensus_result.get("sources", []),
            reasoning=consensus_result.get("reasoning", "Multi-agent consensus"),
            escalated_from=None,
            metadata=consensus_result.get("metadata", {})
        )
    
    # Analysis and routing methods
    
    def _analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> QueryAnalysis:
        """Analyze query to determine processing strategy"""
        query_lower = query.lower()
        
        # Check if vision is needed
        requires_vision = False
        vision_keywords = [
            "image", "photo", "picture", "diagram", "blueprint",
            "drawing", "visual", "show me", "look at", "analyze this"
        ]
        if any(kw in query_lower for kw in vision_keywords):
            requires_vision = True
        
        if context and context.get("images"):
            requires_vision = True
        
        # Check if retrieval is needed
        requires_retrieval = False
        retrieval_keywords = [
            "code", "requirement", "specification", "standard",
            "according to", "what does", "explain", "how to",
            "building code", "regulation", "guideline"
        ]
        if any(kw in query_lower for kw in retrieval_keywords):
            requires_retrieval = True
        
        # Check if consensus is needed
        requires_consensus = False
        critical_keywords = [
            "critical", "important", "safety", "structural",
            "load bearing", "verify", "confirm", "validate"
        ]
        if any(kw in query_lower for kw in critical_keywords):
            requires_consensus = True
        
        # Determine complexity
        if requires_consensus:
            complexity = QueryComplexity.CRITICAL
        elif requires_vision:
            complexity = QueryComplexity.VISUAL
        elif requires_retrieval:
            complexity = QueryComplexity.MODERATE
        elif len(query.split()) > 20:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.SIMPLE
        
        # Estimate processing time
        time_estimates = {
            QueryComplexity.SIMPLE: 2.5,
            QueryComplexity.MODERATE: 3.5,
            QueryComplexity.COMPLEX: 5.0,
            QueryComplexity.VISUAL: 7.0,
            QueryComplexity.CRITICAL: 12.0
        }
        
        return QueryAnalysis(
            query=query,
            complexity=complexity,
            requires_vision=requires_vision,
            requires_retrieval=requires_retrieval,
            requires_consensus=requires_consensus,
            confidence_threshold=self.confidence_thresholds[complexity],
            estimated_processing_time=time_estimates[complexity]
        )
    
    def _determine_starting_level(self, analysis: QueryAnalysis) -> ProcessingLevel:
        """Determine which level to start processing at"""
        if analysis.requires_consensus:
            return ProcessingLevel.LEVEL_4_CONSENSUS
        elif analysis.requires_vision:
            return ProcessingLevel.LEVEL_3_VISION
        elif analysis.requires_retrieval:
            return ProcessingLevel.LEVEL_2_TEXT_RAG
        else:
            return ProcessingLevel.LEVEL_1_FAST_TEXT
    
    def _escalate(self, current_level: ProcessingLevel) -> ProcessingLevel:
        """Escalate to next processing level"""
        if current_level.value < 4:
            return ProcessingLevel(current_level.value + 1)
        return current_level  # Already at max
    
    def _is_confidence_sufficient(
        self,
        result: QueryResult,
        analysis: QueryAnalysis
    ) -> bool:
        """Check if confidence meets threshold"""
        return result.confidence >= analysis.confidence_threshold
    
    def _estimate_confidence(self, answer: str, query: str) -> float:
        """Estimate confidence in answer"""
        # Simplified confidence estimation
        # In production, would use more sophisticated methods
        
        # Long, detailed answers typically more confident
        if len(answer) > 200:
            base_confidence = 0.85
        elif len(answer) > 100:
            base_confidence = 0.75
        else:
            base_confidence = 0.65
        
        # Check for hedging language
        hedging_phrases = [
            "might", "maybe", "possibly", "perhaps",
            "could be", "may be", "uncertain", "not sure"
        ]
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in answer.lower())
        
        # Reduce confidence for hedging
        confidence = base_confidence - (hedge_count * 0.10)
        
        return max(0.0, min(1.0, confidence))
    
    def _estimate_confidence_with_retrieval(
        self,
        answer: str,
        query: str,
        retrieved_docs: List[Dict]
    ) -> float:
        """Estimate confidence when using retrieval"""
        base_confidence = self._estimate_confidence(answer, query)
        
        # Boost confidence if we have good sources
        if retrieved_docs:
            # Check relevance scores
            avg_relevance = sum(
                doc.get("relevance_score", 0.5)
                for doc in retrieved_docs
            ) / len(retrieved_docs)
            
            # Boost by up to 0.15 based on source quality
            boost = avg_relevance * 0.15
            base_confidence += boost
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _analyze_image_cached(
        self,
        image_path: str,
        query: str
    ) -> Dict[str, Any]:
        """Analyze image with caching"""
        if self.cache:
            cached = self.cache.get(image_path, query)
            if cached:
                logger.info(f"Cache hit for {image_path}")
                return cached
        
        # Analyze with vision model
        result = await self.vision.analyze_image(image_path, query)
        
        # Cache result
        if self.cache:
            self.cache.put(image_path, result, query)
        
        return result
    
    def _update_stats(self, result: QueryResult):
        """Update performance statistics"""
        # Update average response time
        n = self.stats["total_queries"]
        old_avg = self.stats["avg_response_time"]
        new_time = result.processing_time
        
        self.stats["avg_response_time"] = (old_avg * (n - 1) + new_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "level_distribution": {
                "level_1_pct": (self.stats["level_1_count"] / total) * 100,
                "level_2_pct": (self.stats["level_2_count"] / total) * 100,
                "level_3_pct": (self.stats["level_3_count"] / total) * 100,
                "level_4_pct": (self.stats["level_4_count"] / total) * 100,
            },
            "escalation_rate": (self.stats["escalations"] / total) * 100
        }


# Convenience functions

async def process_query_progressive(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    llm_engine=None,
    vision_engine=None,
    rag_system=None
) -> QueryResult:
    """
    Convenience function for progressive query processing.
    
    Args:
        query: User's question
        context: Optional context
        llm_engine: Text LLM
        vision_engine: Vision LLM
        rag_system: RAG system
    
    Returns:
        QueryResult with answer
    """
    engine = ProgressiveEnhancementEngine(
        llm_engine=llm_engine,
        vision_engine=vision_engine,
        rag_system=rag_system
    )
    
    return await engine.process_query(query, context)


if __name__ == "__main__":
    print("🚀 Progressive Enhancement Engine Ready")
    print("=" * 60)
    print("\nProcessing Levels:")
    print("  Level 1: Fast text (2-3s) - Simple questions")
    print("  Level 2: Text + RAG (3-4s) - Knowledge retrieval")
    print("  Level 3: Vision (5-8s) - Image analysis")
    print("  Level 4: Consensus (10-15s) - Critical decisions")
    print("\nExpected Distribution:")
    print("  80% queries → Level 1-2 (fast)")
    print("  19% queries → Level 3 (vision)")
    print("  1% queries → Level 4 (consensus)")
    print("\n✅ Smart escalation based on confidence")
    print("✅ Optimal speed/accuracy tradeoff")
