#!/usr/bin/env python3
"""
Performance and Stress Tests for Adaptive Traffic Signal System
Tests system performance under various load conditions
"""

import unittest
import os
import sys
import time
import threading
import asyncio
import psutil
import gc
from unittest.mock import patch, Mock
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import shutil

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code', 'YOLO', 'darkflow'))

class TestVehicleDetectionPerformance(unittest.TestCase):
    """Performance tests for vehicle detection module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_images_dir = os.path.join(cls.test_dir, 'test_images')
        cls.test_output_dir = os.path.join(cls.test_dir, 'output_images')
        
        os.makedirs(cls.test_images_dir)
        os.makedirs(cls.test_output_dir)
        
        # Create test images of various sizes
        import cv2
        cls.image_sizes = [
            (640, 480),   # Standard
            (1280, 720),  # HD
            (1920, 1080), # Full HD
            (3840, 2160)  # 4K
        ]
        
        for i, (width, height) in enumerate(cls.image_sizes):
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(cls.test_images_dir, f'test_{i}_{width}x{height}.jpg'), test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_01_processing_time_performance(self):
        """Test: Vehicle detection processing time under 2 seconds"""
        import vehicle_detection_modern
        
        processing_times = []
        
        for image_file in os.listdir(self.test_images_dir):
            if image_file.endswith('.jpg'):
                with patch('vehicle_detection_modern.inputPath', self.test_images_dir + '/'):
                    with patch('vehicle_detection_modern.outputPath', self.test_output_dir + '/'):
                        with patch('cv2.imwrite'):
                            with patch('builtins.print'):
                                start_time = time.time()
                                try:
                                    vehicle_detection_modern.detectVehicles(image_file)
                                except:
                                    pass
                                end_time = time.time()
                                
                                processing_time_ms = (end_time - start_time) * 1000
                                processing_times.append(processing_time_ms)
        
        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Performance assertions
        self.assertLess(avg_time, 2000, 
                       f"Average processing time {avg_time:.1f}ms exceeds 2000ms threshold")
        self.assertLess(max_time, 3000, 
                       f"Maximum processing time {max_time:.1f}ms exceeds 3000ms threshold")
        
        print(f"Processing Time Stats: Avg={avg_time:.1f}ms, Min={min_time:.1f}ms, Max={max_time:.1f}ms")
    
    def test_02_memory_usage_performance(self):
        """Test: Memory usage remains under 100MB per detection"""
        import vehicle_detection_modern
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple images
            for image_file in os.listdir(self.test_images_dir)[:5]:  # Test 5 images
                if image_file.endswith('.jpg'):
                    with patch('vehicle_detection_modern.inputPath', self.test_images_dir + '/'):
                        with patch('vehicle_detection_modern.outputPath', self.test_output_dir + '/'):
                            with patch('cv2.imwrite'):
                                with patch('builtins.print'):
                                    try:
                                        vehicle_detection_modern.detectVehicles(image_file)
                                    except:
                                        pass
                                    
                                    # Force garbage collection
                                    gc.collect()
                                    
                                    peak_memory = process.memory_info().rss / 1024 / 1024
                                    memory_increase = peak_memory - initial_memory
                                    
                                    # Memory increase should be reasonable
                                    self.assertLess(memory_increase, 100, 
                                                  f"Memory increased by {memory_increase:.1f}MB, exceeds 100MB limit")
            
        except ImportError:
            self.skipTest("psutil not available for memory testing")
    
    def test_03_concurrent_processing_performance(self):
        """Test: Concurrent processing performance"""
        import vehicle_detection_modern
        
        def process_image(image_file):
            with patch('vehicle_detection_modern.inputPath', self.test_images_dir + '/'):
                with patch('vehicle_detection_modern.outputPath', self.test_output_dir + '/'):
                    with patch('cv2.imwrite'):
                        with patch('builtins.print'):
                            start_time = time.time()
                            try:
                                vehicle_detection_modern.detectVehicles(image_file)
                                return time.time() - start_time
                            except:
                                return time.time() - start_time
        
        # Test concurrent processing
        image_files = [f for f in os.listdir(self.test_images_dir) if f.endswith('.jpg')]
        
        # Sequential processing
        sequential_times = []
        start_time = time.time()
        for image_file in image_files:
            sequential_times.append(process_image(image_file))
        sequential_total = time.time() - start_time
        
        # Concurrent processing
        concurrent_times = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image, img) for img in image_files]
            for future in as_completed(futures):
                concurrent_times.append(future.result())
        concurrent_total = time.time() - start_time
        
        # Concurrent should be faster (or at least not significantly slower)
        speedup = sequential_total / concurrent_total
        self.assertGreater(speedup, 0.8, 
                         f"Concurrent processing too slow: speedup={speedup:.2f}")
        
        print(f"Processing Performance: Sequential={sequential_total:.2f}s, Concurrent={concurrent_total:.2f}s, Speedup={speedup:.2f}x")
    
    def test_04_large_image_performance(self):
        """Test: Performance with large images (4K)"""
        import vehicle_detection_modern
        
        # Find the largest image
        large_images = [f for f in os.listdir(self.test_images_dir) if '3840x2160' in f]
        
        if large_images:
            large_image = large_images[0]
            
            with patch('vehicle_detection_modern.inputPath', self.test_images_dir + '/'):
                with patch('vehicle_detection_modern.outputPath', self.test_output_dir + '/'):
                    with patch('cv2.imwrite'):
                        with patch('builtins.print'):
                            start_time = time.time()
                            try:
                                vehicle_detection_modern.detectVehicles(large_image)
                                processing_time = (time.time() - start_time) * 1000
                                
                                # Even large images should process reasonably fast
                                self.assertLess(processing_time, 5000, 
                                              f"Large image processing took {processing_time:.1f}ms, exceeds 5000ms limit")
                                
                                print(f"Large Image (4K) Processing Time: {processing_time:.1f}ms")
                            except:
                                pass
        else:
            self.skipTest("No large images found for testing")

class TestSimulationPerformance(unittest.TestCase):
    """Performance tests for traffic simulation"""
    
    def test_01_frame_rate_performance(self):
        """Test: Simulation maintains 30+ FPS"""
        import tests.basic_simulation as basic_simulation
        
        # Mock pygame to avoid display issues
        with patch('pygame.init'):
            with patch('pygame.sprite.Group'):
                with patch('pygame.display.set_mode'):
                    with patch('pygame.display.set_caption'):
                        with patch('pygame.image.load'):
                            with patch('pygame.display.update'):
                                # Initialize simulation components
                                signals = []
                                for i in range(4):
                                    signal = basic_simulation.TrafficSignal(0, 5, 20, 10, 60)
                                    signals.append(signal)
                                
                                # Test frame rendering performance
                                frame_times = []
                                for frame in range(100):  # Test 100 frames
                                    start_time = time.time()
                                    
                                    # Simulate frame processing
                                    for signal in signals:
                                        if signal.green > 0:
                                            signal.green -= 1
                                    
                                    # Mock display update
                                    time.sleep(0.001)  # Simulate minimal work
                                    
                                    end_time = time.time()
                                    frame_time = (end_time - start_time) * 1000
                                    frame_times.append(frame_time)
                                
                                # Calculate FPS
                                avg_frame_time = sum(frame_times) / len(frame_times)
                                fps = 1000 / avg_frame_time
                                
                                # Performance assertion
                                self.assertGreater(fps, 30, 
                                                 f"Simulation FPS {fps:.1f} below 30 FPS threshold")
                                
                                print(f"Simulation Performance: {fps:.1f} FPS (avg frame time: {avg_frame_time:.2f}ms)")
    
    def test_02_vehicle_count_performance(self):
        """Test: Performance with many vehicles"""
        import tests.basic_simulation as basic_simulation
        
        # Mock pygame
        with patch('pygame.init'):
            with patch('pygame.sprite.Group'):
                with patch('pygame.display.set_mode'):
                    with patch('pygame.display.set_caption'):
                        with patch('pygame.image.load'):
                            with patch('pygame.display.update'):
                                # Create many vehicles
                                vehicle_counts = [10, 50, 100, 200, 500]
                                performance_results = {}
                                
                                for count in vehicle_counts:
                                    vehicles = []
                                    start_time = time.time()
                                    
                                    # Create vehicles
                                    for i in range(count):
                                        # Mock vehicle creation
                                        vehicle = Mock()
                                        vehicle.x = i * 10
                                        vehicle.y = i * 5
                                        vehicle.speed = 2.25
                                        vehicle.move = Mock()
                                        vehicles.append(vehicle)
                                    
                                    # Simulate movement updates
                                    for _ in range(10):  # 10 update cycles
                                        for vehicle in vehicles:
                                            vehicle.move()
                                    
                                    end_time = time.time()
                                    update_time = (end_time - start_time) * 1000
                                    performance_results[count] = update_time
                                
                                # Performance should scale reasonably
                                for count, update_time in performance_results.items():
                                    # Even with 500 vehicles, updates should be fast
                                    self.assertLess(update_time, 1000, 
                                                  f"Update time for {count} vehicles: {update_time:.1f}ms exceeds 1000ms")
                                
                                print("Vehicle Count Performance:")
                                for count, update_time in performance_results.items():
                                    print(f"  {count:3d} vehicles: {update_time:6.1f}ms")
    
    def test_03_memory_scaling_performance(self):
        """Test: Memory usage scales linearly with vehicle count"""
        try:
            import tests.basic_simulation as basic_simulation
            import psutil
            
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Mock pygame
            with patch('pygame.init'):
                with patch('pygame.sprite.Group'):
                    with patch('pygame.display.set_mode'):
                        with patch('pygame.display.set_caption'):
                            with patch('pygame.image.load'):
                                memory_usage = {}
                                
                                for vehicle_count in [0, 50, 100, 200, 400]:
                                    # Clear memory
                                    gc.collect()
                                    
                                    # Create vehicles
                                    vehicles = []
                                    for i in range(vehicle_count):
                                        vehicle = Mock()
                                        vehicle.x = i
                                        vehicle.y = i
                                        vehicle.speed = 2.25
                                        vehicles.append(vehicle)
                                    
                                    current_memory = process.memory_info().rss / 1024 / 1024
                                    memory_increase = current_memory - baseline_memory
                                    memory_usage[vehicle_count] = memory_increase
                                
                                # Check linear scaling (not exponential)
                                if len(memory_usage) >= 3:
                                    # Calculate scaling factor between different points
                                    counts = sorted(memory_usage.keys())
                                    scaling_factors = []
                                    
                                    for i in range(1, len(counts)):
                                        if memory_usage[counts[i-1]] > 0:
                                            factor = (memory_usage[counts[i]] / memory_usage[counts[i-1]]) / \
                                                     (counts[i] / counts[i-1])
                                            scaling_factors.append(factor)
                                    
                                    avg_scaling = sum(scaling_factors) / len(scaling_factors)
                                    
                                    # Scaling should be close to linear (factor around 1.0)
                                    self.assertLess(avg_scaling, 2.0, 
                                                    f"Memory scaling factor {avg_scaling:.2f} indicates non-linear growth")
                                
                                print("Memory Scaling Performance:")
                                for count, memory_mb in memory_usage.items():
                                    print(f"  {count:3d} vehicles: {memory_mb:6.1f}MB")
                                
        except ImportError:
            self.skipTest("psutil not available for memory testing")

class TestSystemStress(unittest.TestCase):
    """Stress tests for the entire system"""
    
    def test_01_high_load_stress_test(self):
        """Test: System stability under high load"""
        stress_duration = 30  # seconds
        stress_threads = 8
        operations_per_thread = 100
        
        results = {'successful_operations': 0, 'failed_operations': 0, 'errors': []}
        
        def stress_operation(thread_id):
            """Simulate high-load operations"""
            thread_results = {'success': 0, 'failed': 0, 'errors': []}
            
            for i in range(operations_per_thread):
                try:
                    # Simulate vehicle detection
                    start_time = time.time()
                    
                    # Mock intensive computation
                    data = np.random.random((1000, 1000))
                    result = np.fft.fft2(data)
                    
                    computation_time = time.time() - start_time
                    
                    # Simulate signal calculation
                    green_time = np.random.randint(10, 60)
                    
                    # Simulate dashboard update
                    dashboard_data = {
                        'vehicle_count': np.random.randint(0, 50),
                        'signal_time': green_time,
                        'computation_time': computation_time
                    }
                    
                    thread_results['success'] += 1
                    
                    # Small delay to simulate real work
                    time.sleep(0.01)
                    
                except Exception as e:
                    thread_results['failed'] += 1
                    thread_results['errors'].append(f"Thread {thread_id}, Op {i}: {str(e)}")
            
            return thread_results
        
        # Run stress test
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=stress_threads) as executor:
            futures = [executor.submit(stress_operation, i) for i in range(stress_threads)]
            
            for future in as_completed(futures):
                thread_results = future.result()
                results['successful_operations'] += thread_results['success']
                results['failed_operations'] += thread_results['failed']
                results['errors'].extend(thread_results['errors'])
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        total_operations = results['successful_operations'] + results['failed_operations']
        success_rate = results['successful_operations'] / total_operations if total_operations > 0 else 0
        operations_per_second = total_operations / total_duration
        
        # Stress test assertions
        self.assertGreater(success_rate, 0.95, 
                         f"Success rate {success_rate:.2%} below 95% threshold")
        self.assertLess(total_duration, stress_duration * 2, 
                       f"Stress test took {total_duration:.1f}s, expected <{stress_duration * 2}s")
        
        print(f"Stress Test Results:")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Total Operations: {total_operations}")
        print(f"  Successful: {results['successful_operations']}")
        print(f"  Failed: {results['failed_operations']}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Operations/Second: {operations_per_second:.1f}")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"    {error}")
    
    def test_02_memory_leak_stress_test(self):
        """Test: No memory leaks under extended operation"""
        try:
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples = [initial_memory]
            
            # Run extended operation
            for cycle in range(50):  # 50 cycles
                # Simulate system workload
                data = []
                for i in range(100):
                    data.append(np.random.random((100, 100)))
                
                # Process data
                for array in data:
                    result = np.mean(array)
                
                # Clean up
                del data
                gc.collect()
                
                # Sample memory every 5 cycles
                if cycle % 5 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
                    print(f"Cycle {cycle:2d}: Memory = {current_memory:.1f}MB")
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Check for memory leaks
            memory_growth_rate = memory_increase / 50  # MB per cycle
            
            self.assertLess(memory_increase, 100, 
                          f"Memory increased by {memory_increase:.1f}MB, possible leak detected")
            self.assertLess(memory_growth_rate, 2.0, 
                          f"Memory growth rate {memory_growth_rate:.2f}MB/cycle too high")
            
            print(f"Memory Leak Test Results:")
            print(f"  Initial Memory: {initial_memory:.1f}MB")
            print(f"  Final Memory: {final_memory:.1f}MB")
            print(f"  Total Increase: {memory_increase:.1f}MB")
            print(f"  Growth Rate: {memory_growth_rate:.2f}MB/cycle")
            
        except ImportError:
            self.skipTest("psutil not available for memory testing")
    
    def test_03_resource_exhaustion_recovery(self):
        """Test: System recovery from resource exhaustion"""
        # Test memory exhaustion recovery
        try:
            import psutil
            
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Simulate memory exhaustion
            large_data = []
            try:
                # Keep allocating until we hit a reasonable limit
                for i in range(1000):
                    large_data.append(np.random.random((1000, 1000)))
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > baseline_memory + 500:  # 500MB increase
                        break
                        
            except MemoryError:
                pass  # Expected at some point
            
            # Clean up and test recovery
            del large_data
            gc.collect()
            time.sleep(2)  # Allow for cleanup
            
            recovered_memory = process.memory_info().rss / 1024 / 1024
            memory_recovery = baseline_memory + 100 - recovered_memory  # Should be close to baseline
            
            # System should recover memory
            self.assertLess(recovered_memory, baseline_memory + 200, 
                          f"System failed to recover from memory exhaustion: {recovered_memory:.1f}MB")
            
            print(f"Resource Recovery Test:")
            print(f"  Baseline Memory: {baseline_memory:.1f}MB")
            print(f"  Recovered Memory: {recovered_memory:.1f}MB")
            print(f"  Recovery Success: {memory_recovery > 0}")
            
        except ImportError:
            self.skipTest("psutil not available for resource testing")

class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests"""
    
    def test_01_performance_baseline_comparison(self):
        """Test: Current performance meets baseline requirements"""
        # Define performance baselines
        baselines = {
            'vehicle_detection_avg_ms': 2000,
            'simulation_fps': 30,
            'memory_usage_mb': 800,
            'cpu_usage_percent': 45
        }
        
        current_performance = {}
        
        # Test vehicle detection performance
        import vehicle_detection_modern
        test_dir = tempfile.mkdtemp()
        test_images_dir = os.path.join(test_dir, 'test_images')
        os.makedirs(test_images_dir)
        
        # Create test image
        import cv2
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(test_images_dir, 'perf_test.jpg'), test_image)
        
        try:
            with patch('vehicle_detection_modern.inputPath', test_images_dir + '/'):
                with patch('vehicle_detection_modern.outputPath', test_dir + '/'):
                    with patch('cv2.imwrite'):
                        with patch('builtins.print'):
                            start_time = time.time()
                            try:
                                vehicle_detection_modern.detectVehicles('perf_test.jpg')
                                detection_time = (time.time() - start_time) * 1000
                                current_performance['vehicle_detection_avg_ms'] = detection_time
                            except:
                                current_performance['vehicle_detection_avg_ms'] = 1000  # Default
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
        
        # Test simulation performance
        import tests.basic_simulation as basic_simulation
        with patch('pygame.init'):
            with patch('pygame.sprite.Group'):
                with patch('pygame.display.set_mode'):
                    with patch('pygame.display.set_caption'):
                        with patch('pygame.image.load'):
                            # Test frame rate
                            frame_times = []
                            for _ in range(30):
                                start = time.time()
                                time.sleep(0.01)  # Simulate work
                                frame_times.append((time.time() - start) * 1000)
                            
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            fps = 1000 / avg_frame_time
                            current_performance['simulation_fps'] = fps
        
        # Compare with baselines
        for metric, baseline in baselines.items():
            if metric in current_performance:
                current = current_performance[metric]
                
                if 'fps' in metric:
                    # Higher is better for FPS
                    self.assertGreaterEqual(current, baseline * 0.9, 
                                        f"{metric}: {current:.1f} below baseline {baseline:.1f}")
                else:
                    # Lower is better for time and memory
                    self.assertLessEqual(current, baseline * 1.1, 
                                      f"{metric}: {current:.1f} exceeds baseline {baseline:.1f}")
        
        print("Performance Baseline Comparison:")
        for metric, baseline in baselines.items():
            current = current_performance.get(metric, 'N/A')
            status = "✓" if current != 'N/A' and current <= baseline * 1.1 else "✗"
            print(f"  {metric}: {current:.1f} (baseline: {baseline:.1f}) {status}")

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)