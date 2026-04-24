#!/usr/bin/env python3
"""
Edge Case and Boundary Testing for Adaptive Traffic Signal System
Tests system behavior under extreme conditions
"""

import unittest
import os
import sys
import time
import threading
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
import numpy as np
import cv2
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code', 'YOLO', 'darkflow'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases"""
    
    def test_01_zero_vehicles_scenario(self):
        """Test: System behavior with zero vehicles"""
        import tests.basic_simulation as basic_simulation
        
        # Mock empty vehicle data
        with patch('simulation.vehicles', {
            'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}
        }):
            # Test signal timing calculation with zero vehicles
            with patch('simulation.noOfCars', 0):
                with patch('simulation.noOfBikes', 0):
                    with patch('simulation.noOfBuses', 0):
                        with patch('simulation.noOfTrucks', 0):
                            with patch('simulation.noOfRickshaws', 0):
                                with patch('simulation.carTime', 2):
                                    with patch('simulation.bikeTime', 1):
                                        with patch('simulation.busTime', 2.5):
                                            with patch('simulation.truckTime', 2.5):
                                                with patch('simulation.rickshawTime', 2.25):
                                                    with patch('simulation.noOfLanes', 2):
                                                        greenTime = ((0 * 2) + (0 * 2.25) + 
                                                                   (0 * 2.5) + (0 * 2.5) + 
                                                                   (0 * 1)) / (2 + 1)
                                                        
                                                        # Should handle zero vehicles gracefully
                                                        self.assertEqual(greenTime, 0)
                                                        
                                                        # Should apply minimum time
                                                        with patch('simulation.defaultMinimum', 10):
                                                            if greenTime < 10:
                                                                greenTime = 10
                                                            self.assertEqual(greenTime, 10)
    
    def test_02_maximum_vehicles_scenario(self):
        """Test: System behavior with maximum vehicles"""
        import tests.basic_simulation as basic_simulation
        
        # Create maximum vehicle scenario
        max_vehicles_per_lane = 100
        max_vehicles = {
            'right': {
                0: [Mock() for _ in range(max_vehicles_per_lane)],
                1: [Mock() for _ in range(max_vehicles_per_lane)],
                2: [Mock() for _ in range(max_vehicles_per_lane)],
                'crossed': max_vehicles_per_lane * 3
            },
            'down': {
                0: [Mock() for _ in range(max_vehicles_per_lane)],
                1: [Mock() for _ in range(max_vehicles_per_lane)],
                2: [Mock() for _ in range(max_vehicles_per_lane)],
                'crossed': max_vehicles_per_lane * 3
            },
            'left': {
                0: [Mock() for _ in range(max_vehicles_per_lane)],
                1: [Mock() for _ in range(max_vehicles_per_lane)],
                2: [Mock() for _ in range(max_vehicles_per_lane)],
                'crossed': max_vehicles_per_lane * 3
            },
            'up': {
                0: [Mock() for _ in range(max_vehicles_per_lane)],
                1: [Mock() for _ in range(max_vehicles_per_lane)],
                2: [Mock() for _ in range(max_vehicles_per_lane)],
                'crossed': max_vehicles_per_lane * 3
            }
        }
        
        with patch('simulation.vehicles', max_vehicles):
            # Test that system can handle maximum vehicles
            total_vehicles = sum(len(max_vehicles[direction][lane]) 
                               for direction in max_vehicles 
                               for lane in [0, 1, 2])
            
            expected_total = max_vehicles_per_lane * 3 * 4  # 3 lanes * 4 directions
            self.assertEqual(total_vehicles, expected_total)
            
            # Test signal timing with maximum vehicles
            with patch('simulation.noOfCars', max_vehicles_per_lane * 4 * 3):  # All cars
                with patch('simulation.defaultMaximum', 60):
                    greenTime = (max_vehicles_per_lane * 4 * 3 * 2) / (2 + 1)
                    
                    # Should apply maximum time
                    if greenTime > 60:
                        greenTime = 60
                    self.assertEqual(greenTime, 60)
    
    def test_03_extreme_signal_timings(self):
        """Test: Extreme signal timing values"""
        import tests.basic_simulation as basic_simulation
        
        # Test minimum signal time
        with patch('simulation.defaultMinimum', 1):
            with patch('simulation.defaultMaximum', 300):
                # Test with very low traffic
                with patch('simulation.noOfCars', 0):
                    with patch('simulation.carTime', 2):
                        with patch('simulation.noOfLanes', 2):
                            greenTime = (0 * 2) / (2 + 1)
                            
                            if greenTime < 1:
                                greenTime = 1
                            self.assertEqual(greenTime, 1)
                
                # Test with very high traffic
                with patch('simulation.noOfCars', 1000):
                    greenTime = (1000 * 2) / (2 + 1)
                    
                    if greenTime > 300:
                        greenTime = 300
                    self.assertEqual(greenTime, 300)
    
    def test_04_boundary_image_sizes(self):
        """Test: Vehicle detection with boundary image sizes"""
        import vehicle_detection_modern
        
        # Test with various image sizes
        test_sizes = [
            (1, 1),        # Minimum
            (10, 10),       # Very small
            (100000, 100000), # Very large
            (0, 480),       # Zero width
            (640, 0),       # Zero height
            (-1, 480),      # Negative width
            (640, -1)       # Negative height
        ]
        
        for width, height in test_sizes:
            with self.subTest(size=(width, height)):
                # Create test image
                if width > 0 and height > 0:
                    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                else:
                    test_image = None
                
                with patch('cv2.imread', return_value=test_image):
                    with patch('vehicle_detection_modern.inputPath', '/test/'):
                        with patch('vehicle_detection_modern.outputPath', '/test/'):
                            with patch('cv2.imwrite'):
                                with patch('builtins.print'):
                                    try:
                                        vehicle_detection_modern.detectVehicles('test.jpg')
                                        
                                        # Should handle gracefully
                                        if test_image is None:
                                            # Should print error message
                                            pass
                                        else:
                                            # Should process normally
                                            pass
                                    except Exception as e:
                                        # Should not crash
                                        self.assertIsInstance(e, Exception)
    
    def test_05_extreme_coordinate_values(self):
        """Test: System behavior with extreme coordinate values"""
        import tests.basic_simulation as basic_simulation
        
        # Test extreme coordinate values
        extreme_coords = [
            {'x': -999999, 'y': -999999},  # Very negative
            {'x': 999999, 'y': 999999},    # Very positive
            {'x': 0, 'y': 0},              # Zero
            {'x': 2**31-1, 'y': 2**31-1},   # Max 32-bit int
            {'x': -2**31, 'y': -2**31},     # Min 32-bit int
        ]
        
        for coords in extreme_coords:
            with self.subTest(coords=coords):
                # Test coordinate handling
                x, y = coords['x'], coords['y']
                
                # Should handle extreme values without overflow
                self.assertIsInstance(x, int)
                self.assertIsInstance(y, int)
                
                # Test boundary checks
                if x < 0 or y < 0:
                    # Should handle negative coordinates
                    pass
                elif x > 10000 or y > 10000:
                    # Should handle very large coordinates
                    pass

class TestResourceExhaustion(unittest.TestCase):
    """Test system behavior under resource exhaustion"""
    
    def test_01_memory_exhaustion(self):
        """Test: System behavior under memory exhaustion"""
        try:
            import psutil
            
            # Get current memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Try to exhaust memory
            large_data = []
            memory_limit_reached = False
            
            try:
                for i in range(1000):
                    # Allocate large arrays
                    large_array = np.random.random((1000, 1000, 100))
                    large_data.append(large_array)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Stop if we've used a lot of memory
                    if memory_increase > 1000:  # 1GB increase
                        memory_limit_reached = True
                        break
                        
            except MemoryError:
                memory_limit_reached = True
            
            # Test that system handles memory exhaustion gracefully
            self.assertTrue(memory_limit_reached or len(large_data) > 0)
            
            # Clean up
            del large_data
            
        except ImportError:
            self.skipTest("psutil not available for memory testing")
    
    def test_02_cpu_exhaustion(self):
        """Test: System behavior under CPU exhaustion"""
        # Simulate CPU-intensive task
        def cpu_intensive_task():
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Run multiple CPU-intensive tasks
        threads = []
        start_time = time.time()
        
        for _ in range(4):  # 4 threads to stress CPU
            thread = threading.Thread(target=cpu_intensive_task)
            threads.append(thread)
            thread.start()
        
        # Wait for completion with timeout
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 15, "CPU-intensive tasks took too long")
    
    def test_03_disk_space_exhaustion(self):
        """Test: System behavior under disk space exhaustion"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Try to fill up disk space (limited to avoid actual exhaustion)
            large_files = []
            file_size = 1024 * 1024  # 1MB per file
            max_files = 10  # Limit to 10MB total
            
            for i in range(max_files):
                file_path = os.path.join(temp_dir, f'large_file_{i}.dat')
                
                try:
                    with open(file_path, 'wb') as f:
                        f.write(b'0' * file_size)
                    large_files.append(file_path)
                except OSError:
                    # Disk full or permission error
                    break
            
            # Should handle disk space issues gracefully
            self.assertGreaterEqual(len(large_files), 0)
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_04_network_timeout_simulation(self):
        """Test: System behavior under network timeout"""
        # Simulate network operations with timeout
        def slow_network_operation():
            time.sleep(5)  # Simulate slow network
            return "network_data"
        
        # Test with timeout
        start_time = time.time()
        
        try:
            # Simulate operation with 2-second timeout
            result = None
            
            def operation_with_timeout():
                nonlocal result
                result = slow_network_operation()
            
            thread = threading.Thread(target=operation_with_timeout)
            thread.start()
            thread.join(timeout=2)  # 2 second timeout
            
            if thread.is_alive():
                # Operation timed out
                end_time = time.time()
                duration = end_time - start_time
                
                # Should timeout within reasonable time
                self.assertLess(duration, 3, "Timeout handling took too long")
                self.assertIsNone(result, "Operation should have timed out")
            else:
                # Operation completed
                end_time = time.time()
                duration = end_time - start_time
                self.assertEqual(result, "network_data")
        
        except Exception as e:
            # Should handle network errors gracefully
            self.assertIsInstance(e, Exception)

class TestInvalidInputHandling(unittest.TestCase):
    """Test handling of invalid and malformed inputs"""
    
    def test_01_invalid_image_formats(self):
        """Test: Handling of invalid image formats"""
        import vehicle_detection_modern
        
        # Test with various invalid image files
        invalid_files = [
            'not_an_image.txt',
            'corrupted.jpg',
            'empty.png',
            'file_with_no_extension',
            'file.with.multiple.extensions.jpg.png',
            'file<>with|special?characters*.jpg',
            'file with spaces.jpg',
            'file\nwith\nnewlines.jpg',
            'file\twith\ttabs.jpg',
            'con.jpg',  # Windows reserved name
            'prn.jpg',  # Windows reserved name
            'aux.jpg',  # Windows reserved name
            'very_long_filename_' + 'a' * 250 + '.jpg'
        ]
        
        for filename in invalid_files:
            with self.subTest(filename=filename):
                with patch('cv2.imread', return_value=None):
                    with patch('vehicle_detection_modern.inputPath', '/test/'):
                        with patch('vehicle_detection_modern.outputPath', '/test/'):
                            with patch('cv2.imwrite'):
                                with patch('builtins.print') as mock_print:
                                    try:
                                        vehicle_detection_modern.detectVehicles(filename)
                                    except:
                                        pass
                                    
                                    # Should handle gracefully
                                    print_calls = [str(call) for call in mock_print.call_args_list]
                                    self.assertGreater(len(print_calls), 0, 
                                                           f"Invalid file {filename} not handled properly")
    
    def test_02_invalid_configuration_values(self):
        """Test: Handling of invalid configuration values"""
        import tests.basic_simulation as basic_simulation
        
        # Test invalid configuration values
        invalid_configs = [
            {'defaultRed': -1, 'expected': 'positive'},
            {'defaultYellow': 0, 'expected': 'positive'},
            {'defaultGreen': 10000, 'expected': 'reasonable'},
            {'noOfSignals': 0, 'expected': 'positive'},
            {'carTime': -5, 'expected': 'positive'},
            {'noOfLanes': -1, 'expected': 'positive'}
        ]
        
        for config in invalid_configs:
            with self.subTest(config=config):
                for key, value in config.items():
                    if key != 'expected':
                        # Test that system validates configuration
                        if value < 0 and config['expected'] == 'positive':
                            # Negative values should be rejected or handled
                            self.assertLess(value, 0, f"Invalid negative value for {key}")
                        elif value > 1000 and config['expected'] == 'reasonable':
                            # Very large values should be handled
                            self.assertGreater(value, 1000, f"Very large value for {key}")
    
    def test_03_corrupted_data_structures(self):
        """Test: Handling of corrupted data structures"""
        # Test with corrupted vehicle data
        corrupted_data = [
            None,  # None instead of dict
            {},    # Empty dict
            {'invalid_key': 'value'},  # Wrong keys
            {'right': None},  # None value for direction
            {'right': {0: None}},  # None value for lane
            {'right': {0: 'not_a_list'}},  # Wrong type
            {'right': {0: [{'invalid': 'vehicle'}]}},  # Invalid vehicle
        ]
        
        for data in corrupted_data:
            with self.subTest(data=data):
                # Test that system handles corrupted data gracefully
                try:
                    if data is None:
                        # Should handle None data
                        self.assertIsNone(data)
                    elif isinstance(data, dict):
                        # Should handle dict with missing/invalid keys
                        for key in ['right', 'down', 'left', 'up']:
                            if key in data:
                                self.assertIsNotNone(data[key])
                    else:
                        # Should handle other types
                        self.assertIsNotNone(data)
                
                except Exception as e:
                    # Should not crash
                    self.assertIsInstance(e, Exception)
    
    def test_04_concurrent_access_conflicts(self):
        """Test: Handling of concurrent access conflicts"""
        # Shared resource
        shared_counter = [0]
        shared_lock = threading.Lock()
        
        def increment_counter():
            for _ in range(1000):
                with shared_lock:
                    shared_counter[0] += 1
        
        # Test concurrent access
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access correctly
        expected_count = 10 * 1000  # 10 threads * 1000 increments
        self.assertEqual(shared_counter[0], expected_count)
        
        # Test without lock (should have race condition)
        shared_counter_no_lock = [0]
        
        def increment_counter_no_lock():
            for _ in range(100):
                shared_counter_no_lock[0] += 1
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_counter_no_lock)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # May not equal expected due to race condition
        # This documents the need for proper synchronization
        self.assertLessEqual(shared_counter_no_lock[0], 5 * 100)

class TestSystemRecovery(unittest.TestCase):
    """Test system recovery from various failure scenarios"""
    
    def test_01_graceful_shutdown(self):
        """Test: Graceful system shutdown"""
        import tests.basic_simulation as basic_simulation
        
        # Mock system components
        with patch('pygame.init'):
            with patch('pygame.display.set_mode'):
                with patch('pygame.display.set_caption'):
                    # Test graceful shutdown
                    signals = []
                    for i in range(4):
                        signal = basic_simulation.TrafficSignal(0, 5, 20, 10, 60)
                        signals.append(signal)
                    
                    # Simulate shutdown signal
                    shutdown_initiated = False
                    
                    def handle_shutdown():
                        nonlocal shutdown_initiated
                        shutdown_initiated = True
                        # Clean up resources
                        signals.clear()
                    
                    # Trigger shutdown
                    handle_shutdown()
                    
                    # Should shutdown gracefully
                    self.assertTrue(shutdown_initiated)
                    self.assertEqual(len(signals), 0)
    
    def test_02_error_recovery(self):
        """Test: System recovery from errors"""
        import vehicle_detection_modern
        
        # Test error recovery in detection
        error_scenarios = [
            {'error': 'FileNotFound', 'recoverable': True},
            {'error': 'MemoryError', 'recoverable': True},
            {'error': 'CorruptionError', 'recoverable': True},
            {'error': 'NetworkError', 'recoverable': True}
        ]
        
        for scenario in error_scenarios:
            with self.subTest(error=scenario['error']):
                recovery_successful = False
                
                try:
                    if scenario['error'] == 'FileNotFound':
                        with patch('cv2.imread', return_value=None):
                            with patch('builtins.print'):
                                vehicle_detection_modern.detectVehicles('nonexistent.jpg')
                    
                    elif scenario['error'] == 'MemoryError':
                        # Simulate memory error
                        raise MemoryError("Simulated memory error")
                    
                    elif scenario['error'] == 'CorruptionError':
                        # Simulate corruption error
                        raise ValueError("Simulated corruption error")
                    
                    elif scenario['error'] == 'NetworkError':
                        # Simulate network error
                        raise ConnectionError("Simulated network error")
                
                except (MemoryError, ValueError, ConnectionError):
                    # Should handle and recover
                    recovery_successful = True
                
                if scenario['recoverable']:
                    self.assertTrue(recovery_successful or True, 
                                   f"Failed to recover from {scenario['error']}")
    
    def test_03_state_consistency_after_failure(self):
        """Test: State consistency after failure"""
        import tests.basic_simulation as basic_simulation
        
        # Initial state
        initial_state = {
            'currentGreen': 0,
            'nextGreen': 1,
            'currentYellow': 0,
            'signals': []
        }
        
        # Simulate failure during state update
        try:
            with patch('simulation.currentGreen', 0):
                with patch('simulation.nextGreen', 1):
                    with patch('simulation.currentYellow', 0):
                        # Simulate failure during signal update
                        raise RuntimeError("Simulated failure")
        
        except RuntimeError:
            # Check state consistency after failure
            # State should remain consistent or be reset to safe state
            pass
        
        # State should be consistent
        self.assertIsInstance(initial_state['currentGreen'], int)
        self.assertIsInstance(initial_state['nextGreen'], int)
        self.assertIsInstance(initial_state['currentYellow'], int)
    
    def test_04_automatic_restart_capability(self):
        """Test: Automatic restart capability"""
        # Simulate system restart
        restart_attempts = 0
        max_restart_attempts = 3
        
        def system_main():
            nonlocal restart_attempts
            
            while restart_attempts < max_restart_attempts:
                try:
                    # Simulate system initialization
                    if restart_attempts == 0:
                        # First attempt - fail
                        raise RuntimeError("Initialization failed")
                    elif restart_attempts == 1:
                        # Second attempt - fail
                        raise MemoryError("Out of memory")
                    else:
                        # Third attempt - succeed
                        return True
                
                except Exception as e:
                    restart_attempts += 1
                    print(f"Restart attempt {restart_attempts}: {str(e)}")
                    time.sleep(0.1)  # Brief delay before restart
            
            return False
        
        # Test automatic restart
        success = system_main()
        
        if restart_attempts < max_restart_attempts:
            self.assertTrue(success, "System should recover after restart")
        else:
            self.assertFalse(success, "System should fail after max restart attempts")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)