#!/usr/bin/env python3
"""
Parallel testing script for A10G GPU-optimized PDF analysis server
Tests concurrent processing capability with multiple simultaneous requests
"""

import asyncio
import aiohttp
import time
import json
from pathlib import Path

async def analyze_pdf(session, pdf_path, request_id):
    """Send single PDF analysis request"""
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='input.pdf', content_type='application/pdf')
            data.add_field('fast', 'false')
            data.add_field('extraction_format', '')
            
            async with session.post('http://localhost/', data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    end_time = time.time()
                    duration = end_time - start_time
                    segments = len(result) if isinstance(result, list) else 0
                    return {
                        'request_id': request_id,
                        'status': 'success',
                        'duration': duration,
                        'segments_found': segments,
                        'response_size': len(str(result))
                    }
                else:
                    end_time = time.time()
                    error_text = await resp.text()
                    return {
                        'request_id': request_id,
                        'status': 'error',
                        'duration': end_time - start_time,
                        'error': f"HTTP {resp.status}: {error_text[:200]}"
                    }
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'status': 'exception',
            'duration': end_time - start_time,
            'error': str(e)
        }

async def run_parallel_test(num_parallel=5):
    """Run parallel PDF analysis test"""
    print(f"🚀 Starting parallel test with {num_parallel} concurrent requests")
    print("📋 Testing A10G GPU server concurrent processing capability")
    print("-" * 60)
    
    # Check if PDF exists
    pdf_path = Path('input.pdf')
    if not pdf_path.exists():
        print("❌ Error: input.pdf not found")
        return
    
    start_total = time.time()
    
    # Create HTTP session with timeout
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Launch parallel requests
        tasks = [
            analyze_pdf(session, pdf_path, i+1) 
            for i in range(num_parallel)
        ]
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_total = time.time()
    total_duration = end_total - start_total
    
    # Analyze results
    print(f"📊 PARALLEL TEST RESULTS ({num_parallel} concurrent requests)")
    print("=" * 60)
    
    successful = 0
    failed = 0
    total_segments = 0
    durations = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i+1}: ❌ Exception: {result}")
            failed += 1
        elif result['status'] == 'success':
            print(f"Request {result['request_id']}: ✅ {result['duration']:.2f}s - {result['segments_found']} segments")
            successful += 1
            total_segments += result['segments_found']
            durations.append(result['duration'])
        else:
            print(f"Request {result['request_id']}: ❌ {result['status']} - {result.get('error', 'Unknown error')}")
            failed += 1
    
    print("-" * 60)
    print(f"📈 PERFORMANCE SUMMARY:")
    print(f"   Total time: {total_duration:.2f} seconds")
    print(f"   Successful: {successful}/{num_parallel}")
    print(f"   Failed: {failed}/{num_parallel}")
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        throughput = successful / total_duration
        
        print(f"   Average processing time: {avg_duration:.2f}s")
        print(f"   Fastest request: {min_duration:.2f}s")
        print(f"   Slowest request: {max_duration:.2f}s")
        print(f"   Throughput: {throughput:.2f} PDFs/second")
        print(f"   Total segments detected: {total_segments}")
        
        # GPU efficiency analysis
        theoretical_sequential = sum(durations)
        efficiency = (theoretical_sequential / total_duration) * 100
        print(f"   Parallel efficiency: {efficiency:.1f}%")
        
        if successful == num_parallel:
            print("🎉 All requests completed successfully!")
            print("✅ A10G GPU server handles parallel load correctly")
        else:
            print("⚠️  Some requests failed - check server logs")
    
    return successful == num_parallel

if __name__ == "__main__":
    # Test different parallel loads
    test_cases = [5, 8, 10]
    
    for num_parallel in test_cases:
        print(f"\n{'='*20} Testing {num_parallel} parallel requests {'='*20}")
        success = asyncio.run(run_parallel_test(num_parallel))
        
        if not success:
            print(f"❌ Test with {num_parallel} parallel requests failed")
            break
        else:
            print(f"✅ Test with {num_parallel} parallel requests passed")
        
        # Small delay between test cases
        time.sleep(2)
    
    print("\n🏁 Parallel testing complete!")