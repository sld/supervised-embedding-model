def main():
    # Contexts: NxV
    # Responses: CxV
    # Right candidates: Nx1
    make_test_tensor(test_filename, candidates_filename)
    # Contexts: 100xV;
    # Responses: 100xV;
    # 40 per row 40*4000 = 160000
    batch
