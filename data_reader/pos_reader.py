from data_reader.base_sequence_reader import BaseSequenceReader


class POSReader(BaseSequenceReader):
    def _read_and_pad_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        blocks = [b for b in raw.split("\n\n") if b]

        sentences = []
        for block in blocks:
            tokens = [tuple(line.split()) for line in block.splitlines() if line.strip()]
            padded = [(self.pad_token, "PAD")] * 2 + tokens + [(self.pad_token, "PAD")] * 2
            sentences.append(padded)
        return sentences