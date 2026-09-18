import unittest

from utils.word_alignment import frame_bounds, token_audio_intervals


class WordAlignmentTests(unittest.TestCase):
    def test_accents_subwords_and_special_tokens(self):
        intervals = token_audio_intervals(
            'A mí café.',
            [('A', 0, 0.4), ('mí', 0.4, 0.4), ('café.', 1, 2)],
            [(0, 0), (0, 1), (2, 4), (5, 7), (7, 9), (9, 10), (0, 0)],
            audio_duration=3,
        )
        self.assertEqual(intervals, [None, (0, 0.4), (0.4, 0.4), (1, 2), (1, 2), (2, 3), None])

    def test_punctuation_does_not_consume_preceding_word(self):
        intervals = token_audio_intervals(
            'Hola! Sí.', [('Hola!', 1, 2), ('Sí.', 3, 4)],
            [(0, 4), (4, 5), (6, 8), (8, 9)],
        )
        self.assertEqual(intervals, [(1, 2), (2, 3), (3, 4), (4, 4)])

    def test_catalan_internal_punctuation(self):
        intervals = token_audio_intervals(
            "l’avi col·legi", [("l’avi", 0, 1), ('col·legi', 1, 2)],
            [(0, 1), (1, 2), (2, 5), (6, 9), (9, 10), (10, 14)],
        )
        self.assertEqual(intervals, [(0, 1)] * 3 + [(1, 2)] * 3)

    def test_repeated_words_and_truncated_last_word(self):
        intervals = token_audio_intervals(
            'sí sí bueno', [('sí', 0, 1), ('sí', 1, 2), ('bueno', 2, 3)],
            [(0, 2), (3, 5), (6, 8)],
        )
        self.assertEqual(intervals, [(0, 1), (1, 2), (2, 3)])

    def test_inconsistent_transcript_fails_instead_of_assigning_gap(self):
        with self.assertRaises(ValueError):
            token_audio_intervals('uno dos', [('dos', 0, 1)], [(0, 3)])
        with self.assertRaises(ValueError):
            token_audio_intervals('uno dos', [('uno', 0, 1)], [(0, 3)])

    def test_short_intervals_have_nonempty_bounded_frames(self):
        self.assertEqual(frame_bounds(0.4, 0.4, 50, 100), (18, 22))
        self.assertEqual(frame_bounds(0, 0, 50, 100), (0, 2))
        self.assertEqual(frame_bounds(2, 2, 50, 100), (97, 100))
        with self.assertRaises(ValueError):
            frame_bounds(0, 1, 50, 0)


if __name__ == '__main__':
    unittest.main()
