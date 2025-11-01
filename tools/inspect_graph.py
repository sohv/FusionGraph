import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


class KnowledgeGraphInspector:
    def __init__(self, graph_store_path: str = "./storage/graph_store.json"):
        """
        Initialize inspector with graph store
        
        Args:
            graph_store_path: Path to graph_store.json file
        """
        self.graph_store_path = Path(graph_store_path)
        self.graph_data = None
        self.load_graph()
    
    def load_graph(self):
        """Load graph data from storage"""
        if not self.graph_store_path.exists():
            print(f" Graph store not found: {self.graph_store_path}")
            return
        
        try:
            with open(self.graph_store_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.graph_data = data.get('graph_dict', {})
            print(f" Loaded graph with {len(self.graph_data)} entities")
        except Exception as e:
            print(f" Error loading graph: {e}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive graph statistics
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.graph_data:
            return {}
        
        total_entities = len(self.graph_data)
        total_relations = sum(len(relations) for relations in self.graph_data.values())
        
        # Analyze relation types
        relation_types = Counter()
        relation_targets = Counter()
        
        for entity, relations in self.graph_data.items():
            for relation_pair in relations:
                if len(relation_pair) >= 2:
                    rel_type = relation_pair[0]
                    target = relation_pair[1]
                    relation_types[rel_type] += 1
                    relation_targets[target] += 1
        
        # Find most connected entities
        entity_connectivity = {
            entity: len(relations) 
            for entity, relations in self.graph_data.items()
        }
        
        most_connected = sorted(
            entity_connectivity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Analyze entity types (heuristic based on patterns)
        entity_types = self._classify_entities()
        
        return {
            'total_entities': total_entities,
            'total_relations': total_relations,
            'avg_relations_per_entity': total_relations / total_entities if total_entities > 0 else 0,
            'top_relation_types': relation_types.most_common(10),
            'most_connected_entities': most_connected,
            'entity_type_distribution': entity_types,
            'unique_relation_types': len(relation_types),
            'unique_targets': len(relation_targets)
        }
    
    def _classify_entities(self) -> Dict[str, int]:
        """
        Classify entities by type based on patterns
        
        Returns:
            Dictionary with entity type counts
        """
        if not self.graph_data:
            return {}
        
        classifications = {
            'documents': 0,
            'concepts': 0,
            'metadata': 0,
            'references': 0,
            'technical_terms': 0,
            'other': 0
        }
        
        for entity in self.graph_data.keys():
            entity_lower = entity.lower()
            
            if any(term in entity_lower for term in ['pdf', 'file_path', 'page_label']):
                classifications['metadata'] += 1
            elif entity_lower.startswith('http'):
                classifications['references'] += 1
            elif any(term in entity_lower for term in ['ai', 'machine learning', 'deep learning', 'algorithm']):
                classifications['technical_terms'] += 1
            elif len(entity) > 50:  # Long text likely concepts/descriptions
                classifications['concepts'] += 1
            elif entity.istitle() or entity.isupper():
                classifications['concepts'] += 1
            else:
                classifications['other'] += 1
        
        return classifications
    
    def find_entity_connections(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Find all connections for a specific entity
        
        Args:
            entity_name: Name of entity to analyze
            max_depth: Maximum depth for connection search
            
        Returns:
            Dictionary with connection information
        """
        if not self.graph_data or entity_name not in self.graph_data:
            return {'error': f"Entity '{entity_name}' not found"}
        
        connections = {
            'direct_relations': [],
            'connected_entities': [],
            'relation_summary': Counter()
        }
        
        # Direct relations
        for relation_pair in self.graph_data[entity_name]:
            if len(relation_pair) >= 2:
                rel_type = relation_pair[0]
                target = relation_pair[1]
                connections['direct_relations'].append({
                    'relation': rel_type,
                    'target': target
                })
                connections['relation_summary'][rel_type] += 1
                connections['connected_entities'].append(target)
        
        # Find reverse connections (entities that point to this one)
        reverse_connections = []
        for other_entity, relations in self.graph_data.items():
            if other_entity != entity_name:
                for relation_pair in relations:
                    if len(relation_pair) >= 2 and relation_pair[1] == entity_name:
                        reverse_connections.append({
                            'source': other_entity,
                            'relation': relation_pair[0],
                            'target': entity_name
                        })
        
        connections['reverse_connections'] = reverse_connections
        connections['total_connections'] = len(connections['direct_relations']) + len(reverse_connections)
        
        return connections
    
    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities containing a term
        
        Args:
            search_term: Term to search for
            limit: Maximum results to return
            
        Returns:
            List of matching entities with connection info
        """
        if not self.graph_data:
            return []
        
        search_lower = search_term.lower()
        matches = []
        
        for entity in self.graph_data.keys():
            if search_lower in entity.lower():
                connections = len(self.graph_data[entity])
                matches.append({
                    'entity': entity,
                    'connections': connections,
                    'preview': entity[:100] + "..." if len(entity) > 100 else entity
                })
        
        # Sort by number of connections (most connected first)
        matches.sort(key=lambda x: x['connections'], reverse=True)
        
        return matches[:limit]
    
    def analyze_document_chunks(self) -> Dict[str, Any]:
        """
        Analyze document-chunk relationships
        
        Returns:
            Analysis of how documents are chunked and connected
        """
        if not self.graph_data:
            return {}
        
        document_analysis = {
            'documents': [],
            'chunks_per_document': Counter(),
            'avg_chunk_size': 0,
            'document_connections': defaultdict(list)
        }
        
        # Find file path entities (documents)
        documents = []
        for entity, relations in self.graph_data.items():
            if 'file_path' in entity.lower() or any('/content/' in str(rel) for rel in relations):
                documents.append(entity)
        
        # Analyze page labels and chunks
        page_entities = [e for e in self.graph_data.keys() if 'page_label' in e.lower()]
        
        document_analysis['documents'] = documents
        document_analysis['total_documents'] = len(documents)
        document_analysis['total_pages'] = len(page_entities)
        
        return document_analysis
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive knowledge graph report
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        if not self.graph_data:
            return " No graph data loaded"
        
        stats = self.get_graph_statistics()
        doc_analysis = self.analyze_document_chunks()
        
        report = f"""
# FusionGraph Knowledge Graph Analysis Report

## Graph Overview
- **Total Entities**: {stats['total_entities']:,}
- **Total Relations**: {stats['total_relations']:,}
- **Average Relations per Entity**: {stats['avg_relations_per_entity']:.2f}
- **Unique Relation Types**: {stats['unique_relation_types']}

## Entity Type Distribution
"""
        
        for entity_type, count in stats['entity_type_distribution'].items():
            percentage = (count / stats['total_entities']) * 100 if stats['total_entities'] > 0 else 0
            report += f"- **{entity_type.title()}**: {count} ({percentage:.1f}%)\n"
        
        report += f"""
## Top Relation Types
"""
        for rel_type, count in stats['top_relation_types']:
            report += f"- **{rel_type}**: {count} occurrences\n"
        
        report += f"""
## Most Connected Entities
"""
        for entity, connections in stats['most_connected_entities']:
            preview = entity[:50] + "..." if len(entity) > 50 else entity
            report += f"- **{preview}**: {connections} connections\n"
        
        report += f"""
## Document Analysis
- **Total Documents**: {doc_analysis['total_documents']}
- **Total Pages**: {doc_analysis['total_pages']}
"""
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"📄 Report saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
        
        return report


def main():
    """CLI interface for knowledge graph inspection"""
    parser = argparse.ArgumentParser(description="Inspect FusionGraph knowledge graph")
    parser.add_argument('--graph-path', '-g', default='./storage/graph_store.json',
                       help='Path to graph_store.json file')
    parser.add_argument('--command', '-c', choices=['stats', 'search', 'entity', 'report'],
                       default='stats', help='Command to execute')
    parser.add_argument('--search-term', '-s', help='Term to search for (with search command)')
    parser.add_argument('--entity-name', '-e', help='Entity name to analyze (with entity command)')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit results')
    
    args = parser.parse_args()
    
    inspector = KnowledgeGraphInspector(args.graph_path)
    
    if args.command == 'stats':
        print("\n Knowledge Graph Statistics")
        print("=" * 50)
        
        stats = inspector.get_graph_statistics()
        
        print(f"Total Entities: {stats['total_entities']:,}")
        print(f"Total Relations: {stats['total_relations']:,}")
        print(f"Avg Relations/Entity: {stats['avg_relations_per_entity']:.2f}")
        print(f"Unique Relation Types: {stats['unique_relation_types']}")
        
        print(f"\n Entity Types:")
        for entity_type, count in stats['entity_type_distribution'].items():
            percentage = (count / stats['total_entities']) * 100 if stats['total_entities'] > 0 else 0
            print(f"  • {entity_type.title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n Top Relation Types:")
        for rel_type, count in stats['top_relation_types'][:5]:
            print(f"  • {rel_type}: {count}")
        
        print(f"\n Most Connected Entities:")
        for entity, connections in stats['most_connected_entities'][:5]:
            preview = entity[:60] + "..." if len(entity) > 60 else entity
            print(f"  • {preview}: {connections} connections")
    
    elif args.command == 'search':
        if not args.search_term:
            print(" Please provide --search-term")
            return
        
        print(f"\n Searching for: '{args.search_term}'")
        print("=" * 50)
        
        results = inspector.search_entities(args.search_term, args.limit)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['preview']}")
                print(f"   Connections: {result['connections']}")
                print()
        else:
            print("No matching entities found")
    
    elif args.command == 'entity':
        if not args.entity_name:
            print(" Please provide --entity-name")
            return
        
        print(f"\n🎯 Analyzing Entity: '{args.entity_name}'")
        print("=" * 50)
        
        connections = inspector.find_entity_connections(args.entity_name)
        
        if 'error' in connections:
            print(connections['error'])
            return
        
        print(f"Total Connections: {connections['total_connections']}")
        print(f"Direct Relations: {len(connections['direct_relations'])}")
        print(f"Reverse Relations: {len(connections['reverse_connections'])}")
        
        print(f"\n Relation Summary:")
        for rel_type, count in connections['relation_summary'].most_common():
            print(f"  • {rel_type}: {count}")
        
        print(f"\n Direct Relations:")
        for rel in connections['direct_relations'][:10]:
            target_preview = rel['target'][:50] + "..." if len(rel['target']) > 50 else rel['target']
            print(f"  • {rel['relation']} → {target_preview}")
        
        if connections['reverse_connections']:
            print(f"\n⬅ Reverse Relations:")
            for rel in connections['reverse_connections'][:5]:
                source_preview = rel['source'][:50] + "..." if len(rel['source']) > 50 else rel['source']
                print(f"  • {source_preview} → {rel['relation']} → {args.entity_name}")
    
    elif args.command == 'report':
        print("\n Generating Knowledge Graph Report")
        print("=" * 50)
        
        report = inspector.generate_report(args.output)
        
        if not args.output:
            print(report)


if __name__ == "__main__":
    main()