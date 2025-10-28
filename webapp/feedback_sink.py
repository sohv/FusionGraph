"""
Feedback Collection Module
Collects and manages user feedback for RAG system improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FeedbackEvent:
    """
    Represents a user feedback event
    """
    id: str
    timestamp: str
    feedback_type: str  # 'helpful', 'not_helpful', 'incorrect', 'missing_info'
    target_id: str  # ID of the node/source being rated
    target_type: str  # 'text_source', 'image_source', 'answer', 'graph_relation'
    user_id: Optional[str] = None
    query: Optional[str] = None
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FeedbackSummary:
    """
    Summary of feedback for analysis
    """
    total_feedback_events: int
    helpful_count: int
    not_helpful_count: int
    avg_rating: float
    top_issues: List[str]
    feedback_by_type: Dict[str, int]
    recent_feedback: List[FeedbackEvent]


class FeedbackCollector:
    """
    Collects and manages user feedback for the RAG system
    """
    
    def __init__(self, storage_path: str = "./storage/feedback.jsonl"):
        """
        Initialize feedback collector
        
        Args:
            storage_path: Path to store feedback events
        """
        self.storage_path = storage_path
        self.ensure_storage_directory()
        self.feedback_cache = []
        self.load_recent_feedback()
    
    def ensure_storage_directory(self):
        """Ensure the storage directory exists"""
        storage_dir = os.path.dirname(self.storage_path)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_feedback_id(self) -> str:
        """Generate unique feedback ID"""
        return f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def add_feedback(self, 
                    target_id: str,
                    feedback_type: str,
                    target_type: str,
                    user_id: Optional[str] = None,
                    query: Optional[str] = None,
                    comment: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new feedback event
        
        Args:
            target_id: ID of the item being rated
            feedback_type: Type of feedback ('helpful', 'not_helpful', etc.)
            target_type: Type of target ('text_source', 'image_source', etc.)
            user_id: Optional user identifier
            query: Optional original query
            comment: Optional user comment
            metadata: Optional additional metadata
            
        Returns:
            Feedback event ID
        """
        feedback_event = FeedbackEvent(
            id=self.generate_feedback_id(),
            timestamp=datetime.now().isoformat(),
            feedback_type=feedback_type,
            target_id=target_id,
            target_type=target_type,
            user_id=user_id,
            query=query,
            comment=comment,
            metadata=metadata or {}
        )
        
        # Save to file
        self.save_feedback_event(feedback_event)
        
        # Add to cache
        self.feedback_cache.append(feedback_event)
        
        # Keep cache limited
        if len(self.feedback_cache) > 100:
            self.feedback_cache = self.feedback_cache[-100:]
        
        return feedback_event.id
    
    def save_feedback_event(self, event: FeedbackEvent):
        """
        Save feedback event to JSONL file
        
        Args:
            event: Feedback event to save
        """
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                json.dump(asdict(event), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Error saving feedback event: {e}")
    
    def load_recent_feedback(self, limit: int = 100) -> List[FeedbackEvent]:
        """
        Load recent feedback events from storage
        
        Args:
            limit: Maximum number of events to load
            
        Returns:
            List of recent feedback events
        """
        events = []
        
        if not os.path.exists(self.storage_path):
            return events
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Get last 'limit' lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in recent_lines:
                    if line.strip():
                        event_data = json.loads(line.strip())
                        events.append(FeedbackEvent(**event_data))
            
            self.feedback_cache = events
            return events
            
        except Exception as e:
            print(f"Error loading feedback events: {e}")
            return events
    
    def get_feedback_summary(self, days: int = 30) -> FeedbackSummary:
        """
        Get feedback summary for analysis
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Feedback summary statistics
        """
        # Load all feedback for the specified period
        all_feedback = self.load_all_feedback()
        
        # Filter by date
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_feedback = []
        
        for event in all_feedback:
            try:
                event_timestamp = datetime.fromisoformat(event.timestamp).timestamp()
                if event_timestamp >= cutoff_date:
                    recent_feedback.append(event)
            except:
                # Include events with invalid timestamps
                recent_feedback.append(event)
        
        # Calculate statistics
        total_events = len(recent_feedback)
        helpful_count = len([e for e in recent_feedback if e.feedback_type == 'helpful'])
        not_helpful_count = len([e for e in recent_feedback if e.feedback_type == 'not_helpful'])
        
        # Calculate average rating (helpful=1, not_helpful=0)
        rated_events = helpful_count + not_helpful_count
        avg_rating = helpful_count / rated_events if rated_events > 0 else 0.5
        
        # Count feedback by type
        feedback_by_type = {}
        for event in recent_feedback:
            feedback_by_type[event.feedback_type] = feedback_by_type.get(event.feedback_type, 0) + 1
        
        # Identify top issues (most common negative feedback)
        top_issues = []
        negative_feedback = [e for e in recent_feedback if e.feedback_type in ['not_helpful', 'incorrect']]
        
        # Group by target_type for issue analysis
        issue_counts = {}
        for event in negative_feedback:
            issue_key = f"{event.target_type}: {event.feedback_type}"
            issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
        
        # Sort and get top issues
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_issues = [f"{issue} ({count} occurrences)" for issue, count in top_issues]
        
        return FeedbackSummary(
            total_feedback_events=total_events,
            helpful_count=helpful_count,
            not_helpful_count=not_helpful_count,
            avg_rating=avg_rating,
            top_issues=top_issues,
            feedback_by_type=feedback_by_type,
            recent_feedback=recent_feedback[-10:]  # Last 10 events
        )
    
    def load_all_feedback(self) -> List[FeedbackEvent]:
        """
        Load all feedback events from storage
        
        Returns:
            List of all feedback events
        """
        events = []
        
        if not os.path.exists(self.storage_path):
            return events
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line.strip())
                        events.append(FeedbackEvent(**event_data))
            
            return events
            
        except Exception as e:
            print(f"Error loading all feedback events: {e}")
            return events
    
    def get_feedback_for_target(self, target_id: str) -> List[FeedbackEvent]:
        """
        Get all feedback for a specific target
        
        Args:
            target_id: ID of the target to get feedback for
            
        Returns:
            List of feedback events for the target
        """
        all_feedback = self.load_all_feedback()
        return [event for event in all_feedback if event.target_id == target_id]
    
    def export_feedback_data(self, output_path: str, format: str = 'json') -> bool:
        """
        Export feedback data for analysis
        
        Args:
            output_path: Path to save exported data
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            all_feedback = self.load_all_feedback()
            
            if format == 'json':
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_events': len(all_feedback),
                    'feedback_events': [asdict(event) for event in all_feedback]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                import csv
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if all_feedback:
                        writer = csv.DictWriter(f, fieldnames=asdict(all_feedback[0]).keys())
                        writer.writeheader()
                        for event in all_feedback:
                            writer.writerow(asdict(event))
            
            return True
            
        except Exception as e:
            print(f"Error exporting feedback data: {e}")
            return False
    
    def analyze_source_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of different source types
        
        Returns:
            Dictionary with performance analysis
        """
        all_feedback = self.load_all_feedback()
        
        source_performance = {}
        
        for event in all_feedback:
            source_type = event.target_type
            if source_type not in source_performance:
                source_performance[source_type] = {
                    'total_feedback': 0,
                    'helpful': 0,
                    'not_helpful': 0,
                    'score': 0.5
                }
            
            source_performance[source_type]['total_feedback'] += 1
            
            if event.feedback_type == 'helpful':
                source_performance[source_type]['helpful'] += 1
            elif event.feedback_type == 'not_helpful':
                source_performance[source_type]['not_helpful'] += 1
        
        # Calculate scores
        for source_type, stats in source_performance.items():
            total_rated = stats['helpful'] + stats['not_helpful']
            if total_rated > 0:
                stats['score'] = stats['helpful'] / total_rated
        
        return source_performance
    
    def get_recent_feedback(self, limit: int = 10) -> List[FeedbackEvent]:
        """
        Get recent feedback events
        
        Args:
            limit: Number of recent events to return
            
        Returns:
            List of recent feedback events
        """
        return self.feedback_cache[-limit:] if self.feedback_cache else []


def main():
    """
    Example usage of FeedbackCollector
    """
    collector = FeedbackCollector()
    
    # Example: Add some feedback
    collector.add_feedback(
        target_id="node_123",
        feedback_type="helpful",
        target_type="text_source",
        query="What is AI?",
        comment="Very relevant information"
    )
    
    # Get summary
    summary = collector.get_feedback_summary()
    print(f"Total feedback events: {summary.total_feedback_events}")
    print(f"Average rating: {summary.avg_rating:.2f}")
    
    print("FeedbackCollector initialized successfully!")


if __name__ == "__main__":
    main()