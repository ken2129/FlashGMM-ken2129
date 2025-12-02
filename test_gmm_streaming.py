#!/usr/bin/env python3
"""
Test script for GMM-based rANS streaming decode
Tests if decode_stream_gmm can decode in multiple chunks
"""

import torch
import numpy as np
import sys

def test_gmm_streaming_decode():
    """Test GMM streaming decode - decode in multiple chunks"""
    
    print("=" * 80)
    print("Testing GMM rANS Streaming Decode")
    print("=" * 80)
    
    # Import the compiled extension
    try:
        from compressai import ans
        print("✓ Successfully imported compressai.ans")
    except ImportError as e:
        print(f"✗ Failed to import compressai.ans: {e}")
        return False
    
    # Test parameters
    num_symbols = 300  # Larger dataset for streaming test
    K = 4
    max_bs_value = 255
    
    print(f"\nTest Configuration:")
    print(f"  Total symbols: {num_symbols}")
    print(f"  Mixture components (K): {K}")
    
    # Generate random symbols and GMM parameters
    np.random.seed(42)
    torch.manual_seed(42)
    
    symbols = torch.randint(-20, 21, (num_symbols,), dtype=torch.int32)
    means = torch.randn(num_symbols, K, dtype=torch.float32)
    scales = torch.abs(torch.randn(num_symbols, K, dtype=torch.float32)) + 0.5
    weights_raw = torch.randn(num_symbols, K, dtype=torch.float32)
    weights = torch.softmax(weights_raw, dim=1)
    
    # Ensure all tensors are contiguous
    symbols = symbols.contiguous()
    means = means.contiguous()
    scales = scales.contiguous()
    weights = weights.contiguous()
    
    print(f"  Symbols range: [{symbols.min().item()}, {symbols.max().item()}]")
    
    # Encode
    print("\n" + "=" * 80)
    print("Encoding all symbols...")
    print("=" * 80)
    
    try:
        encoder = ans.RansEncoder()
        encoded_bytes = encoder.encode_with_indexes_gmm(
            symbols, scales, means, weights, max_bs_value
        )
        print(f"✓ Encoding successful")
        print(f"  Encoded size: {len(encoded_bytes)} bytes")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 1: Decode everything at once with decode_with_indexes_gmm (reference)
    print("\n" + "=" * 80)
    print("Test 1: Decode all at once (reference)")
    print("=" * 80)
    
    try:
        decoder_ref = ans.RansDecoder()
        decoded_all = decoder_ref.decode_with_indexes_gmm(
            encoded_bytes, scales, means, weights, max_bs_value
        )
        print(f"✓ Decoded all {len(decoded_all)} symbols")
        match_ref = torch.equal(symbols, decoded_all)
        print(f"  Matches original: {match_ref}")
    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Decode in 3 chunks using decode_stream_gmm
    print("\n" + "=" * 80)
    print("Test 2: Decode in 3 chunks using decode_stream_gmm")
    print("=" * 80)
    
    chunk_sizes = [100, 100, 100]  # Three chunks of 100 symbols each
    print(f"  Chunk sizes: {chunk_sizes}")
    
    try:
        decoder_stream = ans.RansDecoder()
        decoder_stream.set_stream(encoded_bytes)
        
        decoded_chunks = []
        start_idx = 0
        
        for i, chunk_size in enumerate(chunk_sizes):
            end_idx = start_idx + chunk_size
            print(f"\n  Chunk {i+1}: symbols [{start_idx}:{end_idx}]")
            
            # Get GMM parameters for this chunk
            chunk_scales = scales[start_idx:end_idx].contiguous()
            chunk_means = means[start_idx:end_idx].contiguous()
            chunk_weights = weights[start_idx:end_idx].contiguous()
            
            # Decode this chunk
            chunk_decoded = decoder_stream.decode_stream_gmm(
                chunk_scales, chunk_means, chunk_weights, max_bs_value
            )
            
            print(f"    Decoded {len(chunk_decoded)} symbols")
            print(f"    First 5: {chunk_decoded[:5].tolist()}")
            
            decoded_chunks.append(chunk_decoded)
            start_idx = end_idx
        
        # Concatenate all chunks
        decoded_streamed = torch.cat(decoded_chunks, dim=0)
        print(f"\n✓ Successfully decoded all chunks")
        print(f"  Total decoded: {len(decoded_streamed)} symbols")
        
    except Exception as e:
        print(f"✗ Streaming decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verification
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    # Check if streamed decode matches original
    match_stream = torch.equal(symbols, decoded_streamed)
    print(f"\nStreamed decode matches original: {match_stream}")
    if not match_stream:
        diff_indices = (symbols != decoded_streamed).nonzero(as_tuple=True)[0]
        print(f"  Mismatched indices: {diff_indices.tolist()[:10]}... (showing first 10)")
        print(f"  Total mismatches: {len(diff_indices)}")
        for idx in diff_indices[:5]:
            print(f"    Index {idx}: original={symbols[idx].item()}, decoded={decoded_streamed[idx].item()}")
    
    # Check if streamed decode matches reference decode
    match_ref_stream = torch.equal(decoded_all, decoded_streamed)
    print(f"\nStreamed decode matches reference decode: {match_ref_stream}")
    if not match_ref_stream:
        diff_indices = (decoded_all != decoded_streamed).nonzero(as_tuple=True)[0]
        print(f"  Mismatched indices: {diff_indices.tolist()[:10]}... (showing first 10)")
        print(f"  Total mismatches: {len(diff_indices)}")
    
    # Compare chunks with original
    print("\n" + "=" * 80)
    print("Chunk-by-chunk verification:")
    print("=" * 80)
    start_idx = 0
    all_chunks_match = True
    for i, (chunk_size, chunk_decoded) in enumerate(zip(chunk_sizes, decoded_chunks)):
        end_idx = start_idx + chunk_size
        chunk_original = symbols[start_idx:end_idx]
        chunk_match = torch.equal(chunk_original, chunk_decoded)
        print(f"  Chunk {i+1} [{start_idx}:{end_idx}]: {chunk_match}")
        if not chunk_match:
            all_chunks_match = False
            diff_count = (chunk_original != chunk_decoded).sum().item()
            print(f"    Mismatches: {diff_count}/{chunk_size}")
        start_idx = end_idx
    
    # Final result
    print("\n" + "=" * 80)
    all_match = match_ref and match_stream and match_ref_stream and all_chunks_match
    if all_match:
        print("✓✓✓ ALL STREAMING TESTS PASSED ✓✓✓")
        print("Streaming decode works correctly - can decode in multiple chunks!")
    else:
        print("✗✗✗ STREAMING TESTS FAILED ✗✗✗")
        print(f"  Reference decode: {'PASS' if match_ref else 'FAIL'}")
        print(f"  Streaming vs original: {'PASS' if match_stream else 'FAIL'}")
        print(f"  Streaming vs reference: {'PASS' if match_ref_stream else 'FAIL'}")
        print(f"  All chunks correct: {'PASS' if all_chunks_match else 'FAIL'}")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    success = test_gmm_streaming_decode()
    sys.exit(0 if success else 1)
