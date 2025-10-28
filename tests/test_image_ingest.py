"""
Unit tests for Image Ingestion module
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

# Import the module to test
try:
    from ingest.image_ingest import ImageIngestor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.image_ingest import ImageIngestor


class TestImageIngestor(unittest.TestCase):
    """Test cases for ImageIngestor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(self.test_image_path)
        
        # Mock the heavy models to avoid loading them in tests
        with patch('ingest.image_ingest.SentenceTransformer') as mock_st, \
             patch('ingest.image_ingest.BlipProcessor') as mock_bp, \
             patch('ingest.image_ingest.BlipForConditionalGeneration') as mock_bg:
            
            # Create mock instances
            mock_st.return_value = Mock()
            mock_bp.from_pretrained.return_value = Mock()
            mock_bg.from_pretrained.return_value = Mock()
            
            self.ingestor = ImageIngestor()
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ImageIngestor initialization"""
        self.assertIsNotNone(self.ingestor)
        self.assertIsNotNone(self.ingestor.embedding_model)
        self.assertIsNotNone(self.ingestor.caption_processor)
        self.assertIsNotNone(self.ingestor.caption_model)
        self.assertEqual(len(self.ingestor.image_metadata), 0)
    
    @patch('ingest.image_ingest.pytesseract.image_to_string')
    def test_extract_text_from_image_success(self, mock_ocr):
        """Test successful OCR text extraction"""
        mock_ocr.return_value = "Sample extracted text"
        
        result = self.ingestor.extract_text_from_image(self.test_image_path)
        
        self.assertEqual(result, "Sample extracted text")
        mock_ocr.assert_called_once()
    
    @patch('ingest.image_ingest.pytesseract.image_to_string')
    def test_extract_text_from_image_failure(self, mock_ocr):
        """Test OCR text extraction failure handling"""
        mock_ocr.side_effect = Exception("OCR failed")
        
        result = self.ingestor.extract_text_from_image(self.test_image_path)
        
        self.assertEqual(result, "")
    
    def test_extract_text_from_nonexistent_image(self):
        """Test OCR with non-existent image file"""
        result = self.ingestor.extract_text_from_image("nonexistent.jpg")
        self.assertEqual(result, "")
    
    @patch('ingest.image_ingest.torch.no_grad')
    def test_generate_image_caption(self, mock_no_grad):
        """Test image caption generation"""
        # Mock the caption generation process
        mock_inputs = {'pixel_values': 'mock_tensor'}
        mock_outputs = [Mock()]
        mock_outputs[0] = np.array([1, 2, 3])  # Mock token IDs
        
        self.ingestor.caption_processor.return_value = mock_inputs
        self.ingestor.caption_model.generate.return_value = mock_outputs
        self.ingestor.caption_processor.decode.return_value = "A sample caption"
        
        result = self.ingestor.generate_image_caption(self.test_image_path)
        
        self.assertEqual(result, "A sample caption")
    
    @patch('ingest.image_ingest.cv2.imread')
    @patch('ingest.image_ingest.cv2.findContours')
    def test_detect_objects(self, mock_find_contours, mock_imread):
        """Test object detection"""
        # Mock cv2 operations
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Mock contours
        mock_contour = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        mock_find_contours.return_value = ([mock_contour], None)
        
        with patch('ingest.image_ingest.cv2.contourArea', return_value=6400):  # 80x80 area
            result = self.ingestor.detect_objects(self.test_image_path)
        
        self.assertIsInstance(result, list)
        if result:  # If contours were found
            self.assertIn('id', result[0])
            self.assertIn('bbox', result[0])
            self.assertIn('area', result[0])
    
    def test_create_image_nodes(self):
        """Test creation of image nodes"""
        # Mock the individual methods
        with patch.object(self.ingestor, 'extract_text_from_image', return_value="Sample OCR text"), \
             patch.object(self.ingestor, 'generate_image_caption', return_value="Sample caption"), \
             patch.object(self.ingestor, 'detect_objects', return_value=[{'id': 'obj_1', 'bbox': [10, 10, 50, 50], 'area': 2500, 'type': 'detected_object'}]):
            
            nodes = self.ingestor.create_image_nodes(self.test_image_path)
        
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        
        # Check that we have different types of nodes
        node_types = [node.metadata.get('type') for node in nodes]
        self.assertIn('image', node_types)
    
    def test_create_image_nodes_with_options(self):
        """Test image node creation with different options"""
        with patch.object(self.ingestor, 'extract_text_from_image', return_value=""), \
             patch.object(self.ingestor, 'generate_image_caption', return_value=""), \
             patch.object(self.ingestor, 'detect_objects', return_value=[]):
            
            # Test with OCR disabled
            nodes_no_ocr = self.ingestor.create_image_nodes(
                self.test_image_path, 
                include_ocr=False
            )
            
            # Test with caption disabled
            nodes_no_caption = self.ingestor.create_image_nodes(
                self.test_image_path,
                include_caption=False
            )
            
            # Test with objects disabled
            nodes_no_objects = self.ingestor.create_image_nodes(
                self.test_image_path,
                include_objects=False
            )
        
        # All should return at least the main image node
        self.assertGreater(len(nodes_no_ocr), 0)
        self.assertGreater(len(nodes_no_caption), 0)
        self.assertGreater(len(nodes_no_objects), 0)
    
    def test_process_image_directory(self):
        """Test processing a directory of images"""
        # Create additional test images
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f"test_image_{i}.png")
            test_img = Image.new('RGB', (50, 50), color='blue')
            test_img.save(img_path)
        
        with patch.object(self.ingestor, 'create_image_nodes') as mock_create_nodes:
            mock_create_nodes.return_value = [Mock(), Mock()]  # Return 2 nodes per image
            
            nodes = self.ingestor.process_image_directory(self.temp_dir)
        
        # Should process all 4 images (1 original + 3 new)
        self.assertEqual(mock_create_nodes.call_count, 4)
        self.assertEqual(len(nodes), 8)  # 4 images * 2 nodes each
    
    def test_get_image_metadata(self):
        """Test getting image metadata"""
        # First create some nodes to populate metadata
        with patch.object(self.ingestor, 'extract_text_from_image', return_value="test"), \
             patch.object(self.ingestor, 'generate_image_caption', return_value="test"), \
             patch.object(self.ingestor, 'detect_objects', return_value=[]):
            
            self.ingestor.create_image_nodes(self.test_image_path)
        
        metadata = self.ingestor.get_image_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertGreater(len(metadata), 0)


class TestImageIngestorIntegration(unittest.TestCase):
    """Integration tests that require actual model loading (slower)"""
    
    @unittest.skipIf(not os.environ.get('RUN_INTEGRATION_TESTS'), 
                     "Integration tests skipped (set RUN_INTEGRATION_TESTS=1 to run)")
    def test_real_image_processing(self):
        """Test with real models and image processing"""
        # This test would use actual models - only run if explicitly requested
        ingestor = ImageIngestor()
        
        # Create a test image with text
        test_image = Image.new('RGB', (200, 100), color='white')
        # In a real test, you'd add actual text to the image
        
        temp_path = tempfile.mktemp(suffix='.jpg')
        test_image.save(temp_path)
        
        try:
            nodes = ingestor.create_image_nodes(temp_path)
            self.assertIsInstance(nodes, list)
            self.assertGreater(len(nodes), 0)
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestImageIngestor))
    suite.addTests(loader.loadTestsFromTestCase(TestImageIngestorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)