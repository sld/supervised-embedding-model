def main():
    # Contexts: NxV
    # Responses: CxV
    # Right candidates: Nx1
    # Contexts: 100xV;
    # Responses: 100xV;
    # 40 per row 40*4000 = 160000
    # CandidatesTensor
    # TestTensor
    # Score in TestTensor
    # If score in Candidate > Test => 0 : else_in_the_end 1
    test_tensor = make_tensor(test_filename, vocab_filename)
    candidates_tensor = make_tensor(candidates_filename, vocab_filename)
    # TODO: move to tensorflow model?
    for row in test_tensor:
        test_score = sess.run(
            model.f,
            feed_dict={model.context_batch: [row[0]], model.response_batch: [row[1]]}
        )
        print(test_score)
        test_score = test_score[0][0]

        for batch in batch_iter(candidates_tensor, 256):
            candidate_responses = batch[0][0]
            context = row[0]
            context_batch = np.repeat(context, candidate_responses.shape[0], axis=0)

            print(candidate_responses)
            print(context_batch)

            scores = sess.run(
                model.f,
                feed_dict={model.context_batch: [row[0]], model.response_batch: [row[1]]}
            )
            print(scores)
            return
