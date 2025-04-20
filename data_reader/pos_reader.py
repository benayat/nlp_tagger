from data_reader.base_sequence_reader import BaseSequenceReader
from data_reader.util import read_file_lines, process_sentences, process_sentences_for_test


class POSReader(BaseSequenceReader):
    def _read_and_pad_sentences(self, file_path, is_test=False):
        blocks = read_file_lines(file_path)
        return process_sentences(blocks, self.pad_token, task='pos') if not is_test else process_sentences_for_test(blocks, self.pad_token)