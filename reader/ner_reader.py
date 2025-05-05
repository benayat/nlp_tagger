from reader.base_reader import BaseReader
from reader.utils import convert_blocks_to_sentences, read_blocks


class NerReader(BaseReader):
    def _read_file(self, path_to_file: str, is_test: bool = False):
        return convert_blocks_to_sentences(
            read_blocks(path_to_file), self.padding_token, skip_docstart_lines=True,
            is_test=is_test
        )