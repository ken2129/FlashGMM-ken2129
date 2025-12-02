#!/usr/bin/env python3
"""
Test script for GMM-based rANS encoding/decoding
Tests both decode_with_indexes_gmm and decode_stream_gmm
"""

import torch
import numpy as np
import sys

def test_gmm_codec():
    """Test GMM encoding and decoding with both methods"""
    
    print("=" * 80)
    print("Testing GMM rANS Codec")
    print("=" * 80)
    
    # Import the compiled extension
    try:
        from compressai import ans
        print("✓ Successfully imported compressai.ans")
        print(f"  Module path: {ans.__file__}")
        print(f"  Has decode_stream_gmm: {hasattr(ans.RansDecoder, 'decode_stream_gmm')}")
    except ImportError as e:
        print(f"✗ Failed to import compressai.ans: {e}")
        print("Please run: python setup.py install")
        return False
    
    # Test parameters
    num_symbols = 100
    K = 4  # Number of mixture components
    max_bs_value = 255  # Maximum value for binary search
    
    print(f"\nTest Configuration:")
    print(f"  Number of symbols: {num_symbols}")
    print(f"  Mixture components (K): {K}")
    print(f"  Max binary search value: {max_bs_value}")
    
    # Generate random symbols (integers in a reasonable range)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Symbols: random integers between -20 and 20
    symbols = torch.randint(-20, 21, (num_symbols,), dtype=torch.int32)
    print(f"\n✓ Generated {num_symbols} random symbols")
    print(f"  Range: [{symbols.min().item()}, {symbols.max().item()}]")
    print(f"  First 10 symbols: {symbols[:10].tolist()}")
    
    # Create GMM parameters (standard normal mixture)
    # Each symbol has its own GMM parameters (num_symbols x K)
    
    # Means: sample from standard normal for each component
    means = torch.randn(num_symbols, K, dtype=torch.float32)
    
    # Scales: positive values, sample from exponential-like distribution
    scales = torch.abs(torch.randn(num_symbols, K, dtype=torch.float32)) + 0.5
    
    # Weights: should sum to 1 for each symbol, use softmax
    weights_raw = torch.randn(num_symbols, K, dtype=torch.float32)
    weights = torch.softmax(weights_raw, dim=1)
    
    print(f"\n✓ Generated GMM parameters")
    print(f"  Means shape: {means.shape}")
    print(f"  Scales shape: {scales.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Example weights sum: {weights[0].sum().item():.6f} (should be ~1.0)")
    
    # Ensure all tensors are contiguous
    symbols = symbols.contiguous()
    means = means.contiguous()
    scales = scales.contiguous()
    weights = weights.contiguous()
    
    # Encode
    print("\n" + "=" * 80)
    print("Encoding...")
    print("=" * 80)
    
    try:
        encoder = ans.RansEncoder()
        encoded_bytes = encoder.encode_with_indexes_gmm(
            symbols, scales, means, weights, max_bs_value
        )
        print(f"✓ Encoding successful")
        print(f"  Original symbols size: {num_symbols * 4} bytes (int32)")
        print(f"  Encoded size: {len(encoded_bytes)} bytes")
        print(f"  Compression ratio: {(num_symbols * 4) / len(encoded_bytes):.2f}x")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Decode using decode_with_indexes_gmm
    print("\n" + "=" * 80)
    print("Decoding with decode_with_indexes_gmm...")
    print("=" * 80)
    
    try:
        decoder1 = ans.RansDecoder()
        decoded_symbols1 = decoder1.decode_with_indexes_gmm(
            encoded_bytes, scales, means, weights, max_bs_value
        )
        print(f"✓ Decoding successful (decode_with_indexes_gmm)")
        print(f"  Decoded shape: {decoded_symbols1.shape}")
        print(f"  Decoded dtype: {decoded_symbols1.dtype}")
        print(f"  First 10 decoded: {decoded_symbols1[:10].tolist()}")
    except Exception as e:
        print(f"✗ Decoding failed (decode_with_indexes_gmm): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Decode using decode_stream_gmm
    print("\n" + "=" * 80)
    print("Decoding with decode_stream_gmm...")
    print("=" * 80)
    
    try:
        decoder2 = ans.RansDecoder()
        decoder2.set_stream(encoded_bytes)
        decoded_symbols2 = decoder2.decode_stream_gmm(
            scales, means, weights, max_bs_value
        )
        print(f"✓ Decoding successful (decode_stream_gmm)")
        print(f"  Decoded shape: {decoded_symbols2.shape}")
        print(f"  Decoded dtype: {decoded_symbols2.dtype}")
        print(f"  First 10 decoded: {decoded_symbols2[:10].tolist()}")
    except Exception as e:
        print(f"✗ Decoding failed (decode_stream_gmm): {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    # Check if decode_with_indexes_gmm matches original
    match1 = torch.equal(symbols, decoded_symbols1)
    print(f"\ndecode_with_indexes_gmm matches original: {match1}")
    if not match1:
        diff_indices = (symbols != decoded_symbols1).nonzero(as_tuple=True)[0]
        print(f"  Mismatched indices: {diff_indices.tolist()[:10]}... (showing first 10)")
        print(f"  Total mismatches: {len(diff_indices)}")
        for idx in diff_indices[:5]:
            print(f"    Index {idx}: original={symbols[idx].item()}, decoded={decoded_symbols1[idx].item()}")
    
    # Check if decode_stream_gmm matches original
    match2 = torch.equal(symbols, decoded_symbols2)
    print(f"\ndecode_stream_gmm matches original: {match2}")
    if not match2:
        diff_indices = (symbols != decoded_symbols2).nonzero(as_tuple=True)[0]
        print(f"  Mismatched indices: {diff_indices.tolist()[:10]}... (showing first 10)")
        print(f"  Total mismatches: {len(diff_indices)}")
        for idx in diff_indices[:5]:
            print(f"    Index {idx}: original={symbols[idx].item()}, decoded={decoded_symbols2[idx].item()}")
    
    # Check if both decode methods match each other
    match3 = torch.equal(decoded_symbols1, decoded_symbols2)
    print(f"\ndecode_with_indexes_gmm matches decode_stream_gmm: {match3}")
    if not match3:
        diff_indices = (decoded_symbols1 != decoded_symbols2).nonzero(as_tuple=True)[0]
        print(f"  Mismatched indices: {diff_indices.tolist()[:10]}... (showing first 10)")
        print(f"  Total mismatches: {len(diff_indices)}")
        for idx in diff_indices[:5]:
            print(f"    Index {idx}: method1={decoded_symbols1[idx].item()}, method2={decoded_symbols2[idx].item()}")
    
    # Final result
    print("\n" + "=" * 80)
    all_match = match1 and match2 and match3
    if all_match:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Both decoding methods correctly recover the original symbols!")
    else:
        print("✗✗✗ TESTS FAILED ✗✗✗")
        if match1 and match2:
            print("Both methods decode correctly but produce different results (unexpected!)")
        elif not (match1 or match2):
            print("Both methods failed to decode correctly")
        else:
            print(f"decode_with_indexes_gmm: {'PASS' if match1 else 'FAIL'}")
            print(f"decode_stream_gmm: {'PASS' if match2 else 'FAIL'}")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    success = test_gmm_codec()
    sys.exit(0 if success else 1)
