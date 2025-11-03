'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import os
import re

__all__ = ['SingleFileDataLoader', 'MultiFileDataLoader', 'DataLoader', 'DarpaE3DataLoader']

class SingleFileDataLoader:
    def __init__(self, filepath: str):
        '''
        iterates through the lines of a file.

        Args:
            filepath: str, the path of the file.
        '''
        self.file = open(filepath, 'r', buffering=8192, encoding='utf-8')
        print('Loading {} ...'.format(self.file.name)) 
        
    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()

    def __getitem__(self, index: int) -> str:
        '''
        Get the line at the specified index.
        Enumerates the file from the beginning to the specified index.
        '''
        current_position = self.file.tell()  # Save current position
        self.file.seek(0)
        for i, line in enumerate(self.file):
            if i == index:
                self.file.seek(current_position)  # Restore position
                return line.strip()
        self.file.seek(current_position)
        raise IndexError('Index out of range')
    
class MultiFileDataLoader:
    '''
    iterates through the lines of multiple files.

    Args:
        filepaths: list of str, the paths of the files.
    '''
    def __init__(self, filepaths: str):
        self.files = iter([open(filepath, 'r', buffering=8192, encoding='utf-8') for filepath in filepaths])
        print('Loading {} ...'.format(self.current_file.name))

    def __iter__(self):
        self.current_file = next(self.files)
        return self

    def __next__(self) -> str:
        line = self.current_file.readline()
        while not line:
            self.current_file.close()
            self.current_file = next(self.files)
            print('Loading {} ...'.format(self.current_file.name))
            line = self.current_file.readline()
        return line.strip()
    
class DataLoader:
    def __init__(self, _path: str):
        '''
        Initialize the data loader.
        If the path is a file, use a SingleFileDataLoader.
        If the path is a directory, use a MultiFileDataLoader.
        '''
        def extract_path_number(filepath: str) -> int:
            # Function to extract the number from the file name
            match = re.search(r'\d+$', filepath)
            return int(match.group()) if match else 0

        self.filepaths = []
        if os.path.isfile(_path):
            self.filepaths.append(_path)
            self.loader = SingleFileDataLoader(_path)
        elif os.path.isdir(_path):
            for file in os.listdir(_path):
                filesplit = file.split('.')
                if filesplit[0] == 'json' or filesplit[1] == 'json':
                    self.filepaths.append(os.path.join(_path, file))
            self.filepaths = sorted(self.filepaths, key=extract_path_number)
            self.loader = MultiFileDataLoader(self.filepaths)
        else:
            raise ValueError('Invalid path')
        
    def __iter__(self):
        return iter(self.loader)

class DarpaE3DataLoader(DataLoader):
    def __init__(self, path):
        super().__init__(path)
