"""
Unit tests for Visual RAG Pipeline
"""

import unittest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
try:
    from pipeline.visual_rag import VisualRAGPipeline, VisualRAGResult
    from ingest.image_ingest import ImageIngestor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.visual_rag import VisualRAGPipeline, VisualRAGResult
    from ingest.image_ingest import ImageIngestor


class TestVisualRAGPipeline(unittest.TestCase):
    """Test cases for VisualRAGPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the KnowledgeGraphIndex
        self.mock_kg_index = Mock()
        self.mock_kg_index.storage_context.graph_store = Mock()
        self.mock_kg_index.get_networkx_graph.return_value = Mock()
        
        # Mock the ImageIngestor
        with patch('pipeline.visual_rag.ImageIngestor') as mock_ingestor_class:
            mock_ingestor_class.return_value = Mock()
            self.pipeline = VisualRAGPipeline(self.mock_kg_index)
    
    def test_initialization(self):
        """Test VisualRAGPipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.kg_index, self.mock_kg_index)
        self.assertIsNotNone(self.pipeline.image_ingestor)
        self.assertEqual(self.pipeline.similarity_threshold, 0.7)
        self.assertEqual(len(self.pipeline.image_node_mapping), 0)
    
    def test_initialization_with_custom_ingestor(self):
        """Test initialization with custom image ingestor"""
        custom_ingestor = Mock()
        pipeline = VisualRAGPipeline(self.mock_kg_index, custom_ingestor)
        
        self.assertEqual(pipeline.image_ingestor, custom_ingestor)
    
    @patch('os.path.isdir')
    def test_add_images_to_kg_single_image(self, mock_isdir):
        """Test adding a single image to knowledge graph"""
        mock_isdir.return_value = False
        
        # Mock image node creation
        mock_nodes = [Mock(), Mock()]
        self.pipeline.image_ingestor.create_image_nodes.return_value = mock_nodes
        
        # Mock node insertion
        self.pipeline.kg_index.insert_nodes = Mock()
        
        result = self.pipeline.add_images_to_kg("test_image.jpg")
        
        self.assertEqual(result, 2)
        self.pipeline.image_ingestor.create_image_nodes.assert_called_once_with("test_image.jpg")
        self.assertEqual(self.pipeline.kg_index.insert_nodes.call_count, 2)
    
    @patch('os.path.isdir')
    def test_add_images_to_kg_directory(self, mock_isdir):
        """Test adding images from a directory"""
        mock_isdir.return_value = True
        
        # Mock directory processing
        mock_nodes = [Mock(), Mock(), Mock()]
        self.pipeline.image_ingestor.process_image_directory.return_value = mock_nodes
        
        # Mock node insertion
        self.pipeline.kg_index.insert_nodes = Mock()
        
        result = self.pipeline.add_images_to_kg("test_directory/")
        
        self.assertEqual(result, 3)
        self.pipeline.image_ingestor.process_image_directory.assert_called_once_with("test_directory/")
        self.assertEqual(self.pipeline.kg_index.insert_nodes.call_count, 3)
    
    def test_add_images_to_kg_list(self):
        """Test adding a list of images"""
        image_list = ["img1.jpg", "img2.png", "img3.gif"]
        
        # Mock image node creation for each image
        mock_nodes_per_image = [Mock(), Mock()]
        self.pipeline.image_ingestor.create_image_nodes.return_value = mock_nodes_per_image
        
        # Mock node insertion
        self.pipeline.kg_index.insert_nodes = Mock()
        
        result = self.pipeline.add_images_to_kg(image_list)
        
        self.assertEqual(result, 6)  # 3 images * 2 nodes each
        self.assertEqual(self.pipeline.image_ingestor.create_image_nodes.call_count, 3)
        self.assertEqual(self.pipeline.kg_index.insert_nodes.call_count, 6)
    
    def test_query_with_visual_context(self):
        """Test querying with visual context"""
        # Mock query engine
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.source_nodes = [
            Mock(id_='text_1', text='Sample text', metadata={'type': 'text'}, score=0.8),
            Mock(id_='img_1', text='Image caption', metadata={'type': 'image_caption'}, score=0.7)
        ]
        mock_query_engine.query.return_value = mock_response
        
        self.pipeline.kg_index.as_query_engine.return_value = mock_query_engine
        
        # Mock graph context extraction
        with patch.object(self.pipeline, '_extract_graph_context') as mock_extract:
            mock_extract.return_value = {'nodes': {}, 'edges': []}
            
            result = self.pipeline.query_with_visual_context("What is AI?")
        
        self.assertIsInstance(result, VisualRAGResult)
        self.assertIsNotNone(result.answer)
        self.assertEqual(len(result.text_sources), 1)
        self.assertEqual(len(result.image_sources), 1)
        self.assertGreater(result.confidence_score, 0)
    
    def test_extract_graph_context(self):
        """Test graph context extraction"""
        # Mock response with source nodes
        mock_response = Mock()
        mock_source_node = Mock(id_='node_1', text='Sample text', metadata={'type': 'text'})
        mock_response.source_nodes = [mock_source_node]
        
        # Mock networkx graph
        mock_graph = Mock()
        mock_graph.nodes = {'node_1': {'text': 'Sample text'}}
        mock_graph.edges = [('node_1', 'node_2')]
        mock_graph.neighbors.return_value = ['node_2']
        mock_graph.edges.__getitem__.return_value = {'relation': 'connected_to'}
        
        self.pipeline.kg_index.get_networkx_graph.return_value = mock_graph
        
        result = self.pipeline._extract_graph_context("test query", mock_response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('nodes', result)
        self.assertIn('edges', result)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        # Test with no sources
        score = self.pipeline._calculate_confidence_score([], [])
        self.assertEqual(score, 0.0)
        
        # Test with text sources only
        text_sources = [{'score': 0.8}, {'score': 0.9}]
        score = self.pipeline._calculate_confidence_score(text_sources, [])
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
        
        # Test with mixed sources (should get multimodal bonus)
        image_sources = [{'score': 0.7}]
        score = self.pipeline._calculate_confidence_score(text_sources, image_sources)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_get_image_metadata(self):
        """Test getting image metadata"""
        # Set up some mock metadata
        self.pipeline.image_node_mapping = {'img_1': {'node_id': 'node_1', 'image_path': 'test.jpg'}}
        self.pipeline.image_ingestor.get_image_metadata.return_value = {'img_1': 'metadata'}
        
        result = self.pipeline.get_image_metadata()
        
        self.assertIsInstance(result, dict)
        self.assertIn('image_ingestor_metadata', result)
        self.assertIn('image_node_mapping', result)
        self.assertIn('total_images', result)
        self.assertEqual(result['total_images'], 1)
    
    def test_visualize_multimodal_result(self):
        """Test multimodal result visualization preparation"""
        # Create a mock VisualRAGResult
        mock_result = VisualRAGResult(
            answer="Test answer",
            text_sources=[{'id': 'text_1', 'text': 'Sample text', 'metadata': {}}],
            image_sources=[{'id': 'img_1', 'text': 'Caption', 'metadata': {'image_path': 'test.jpg'}}],
            graph_context={'nodes': {'node_1': {}}, 'edges': []},
            confidence_score=0.85,
            provenance={'text_node_ids': ['text_1'], 'image_node_ids': ['img_1']}
        )
        
        result = self.pipeline.visualize_multimodal_result(mock_result)
        
        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertIn('source_breakdown', result)
        self.assertIn('graph_visualization', result)
        self.assertIn('image_previews', result)
        
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(len(result['image_previews']), 1)


class TestVisualRAGResult(unittest.TestCase):
    """Test cases for VisualRAGResult dataclass"""
    
    def test_visual_rag_result_creation(self):
        """Test creating a VisualRAGResult"""
        result = VisualRAGResult(
            answer="Test answer",
            text_sources=[],
            image_sources=[],
            graph_context={},
            confidence_score=0.75,
            provenance={}
        )
        
        self.assertEqual(result.answer, "Test answer")
        self.assertEqual(len(result.text_sources), 0)
        self.assertEqual(len(result.image_sources), 0)
        self.assertEqual(result.confidence_score, 0.75)
        self.assertIsInstance(result.graph_context, dict)
        self.assertIsInstance(result.provenance, dict)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVisualRAGPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualRAGResult))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)