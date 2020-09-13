
class Segmenter:

    def __init__(self, corpus_file_path):
        self.word_dictory = self.extract_dictionary(corpus_file_path)


    def extract_dictionary(self, corpus_file_path):
        print("Building dictionary...")
        word_dictionary = {}
        corpus = open(corpus_file_path, encoding='utf-8')
        for line in corpus:
            for word in line.split():
                if self.is_all_chinese(word):
                    if word in word_dictionary.keys():
                        word_dictionary[word] += 1
                    else:
                        word_dictionary[word] = 1
        print("Done...")
        return word_dictionary



    def is_all_chinese(self, strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True


    def has_no_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return False
        return True


    def get_DAG(self, line, max_word_len):
        DAG = {}
        for left in range(len(line)):
            possible_end_index = []
            right = left + 1
            while right < len(line):
                sub_str = line[left:right]
                
                if self.is_all_chinese(sub_str) and sub_str in self.word_dictory.keys():
                    possible_end_index.append(right)
                    right += 1
                    
                elif self.has_no_chinese(sub_str):
                    while right < len(line) and self.has_no_chinese(line[left:right]):
                        right += 1
                    possible_end_index.append(right - 1)
                    break
                    
                else:
                    break
                    
            DAG[left] = possible_end_index
        return DAG


    def get_DAG_reverse(self, line, max_word_len):
        DAG = {}
        right = len(line)
        while right > 0:
            possible_begin_index = []
            left = right - 1
            while left >= 0:
                sub_str = line[left:right]
                if self.is_all_chinese(sub_str) and sub_str in self.word_dictory.keys():
                    possible_begin_index.append(left)
                    left -= 1
                
                elif self.has_no_chinese(sub_str):
                    while left >= 0 and self.has_no_chinese(line[left:right]):
                        left -= 1
                    possible_begin_index.append(left + 1)
                
                else:
                    break
            DAG[right] = possible_begin_index
            right -= 1
        return DAG


    def get_max_match_route(self, DAG):
        route = {}
        i = 0
        for i in range(len(DAG)):
            if DAG[i] != []:
                route[i] = max(DAG[i])
            else:
                j = i
                while j < len(DAG) and DAG[j] == [] :
                    j += 1
                route[i] = j
        return route


    def get_max_match_route_reverse(self, DAG):
        route = {}
        i = 0
        right = len(DAG)
        while right > 0:
            if DAG[right] != []:
                left = min(DAG[right])
                if left not in route.keys():
                    route[left] = right
                right = left
            else:
                left = right - 1
                while left > 0 and DAG[left] == []:
                    left -= 1
                if left not in route.keys():
                    route[left] = right
                right = left
        return route


    def get_min_match_route(self, DAG):
        route = {}
        i = 0
        for i in range(len(DAG)):
            if DAG[i] != []:
                if (i + 1) in DAG[i] and DAG[i] != [i + 1]:
                    DAG[i].remove(i + 1)

                route[i] = min(DAG[i])
            else:
                j = i
                while(j < len(DAG) and DAG[j] == []):
                    j += 1
                route[i] = j
        return route


    def get_min_match_route_reverse(self, DAG):
        route = {}
        i = 0
        right = len(DAG)
        while right > 0:
            if DAG[right] != []:
                if (right - 1) in DAG[right] and DAG[right] != [right - 1]:
                    DAG[right].remove(right - 1)
                    
                left = max(DAG[right])
                if left not in route.keys():
                    route[left] = right
                right = left
            else:
                left = right - 1
                while left > 0 and DAG[left] == []:
                    left -= 1
                if left not in route.keys():
                    route[left] = right
                right = left
        return route


    def get_max_probability_route(self, DAG, line):
        route = {}
        temp = {}
        N = len(line)
        temp[N - 1] = (1, N)
        left = N - 2
        while left >= 0:
            while left >= 0 and DAG[left] == []:
                temp[left] = (0, left+1)
                left -= 1
            if left >= 0:
                temp[left] = max( ((self.word_dictory.get(line[left:x]) or 1) / len(self.word_dictory) 
                             * temp[x][0], x) for x in DAG[left])
            left -= 1
        
        left = 0
        while left < N:
            route[left] = temp[left][1]
            left = temp[left][1]
        return route


    def segment(self, line, route):
        words = []
        i = 0
        while i < len(line):
            words.append(line[i: route[i]])
            i = route[i]
        return words


    def single_character_num(self, segment):
        number = 0
        for word in segment:
            if len(word) == 1:
                number += 1
        return number


    def longest_match_segment(self, line, max_word_len):
        DAG = self.get_DAG(line, max_word_len)
        route = self.get_max_match_route(DAG)
        return self.segment(line, route)


    def shortest_match_segment(self, line, max_word_len):
        DAG = self.get_DAG(line, max_word_len)
        route = self.get_min_match_route(DAG)
        return self.segment(line, route)


    def longest_match_segment_reverse(self, line, max_word_len):
        DAG = self.get_DAG_reverse(line, max_word_len)
        route = self.get_max_match_route_reverse(DAG)
        return self.segment(line, route)


    def shortest_match_segment_reverse(self, line, max_word_len):
        DAG = self.get_DAG_reverse(line, max_word_len)
        route = self.get_min_match_route_reverse(DAG)
        return self.segment(line, route)


    def bidirectional_match_segment(self, line, max_word_len):
        segment = self.longest_match_segment(line, max_word_len)
        segment_reverse = self.longest_match_segment_reverse(line, max_word_len)
        if len(segment) == len(segment_reverse):
            single_character_number = self.single_character_num(segment)
            single_character_number_reverse = self.single_character_num(segment_reverse)
            if single_character_number < single_character_number_reverse:
                return segment
            else:
                return segment_reverse
        elif len(segment) > len(segment_reverse):
            return segment_reverse
        else:
            return segment


    def most_probability_segment(self, line, max_word_len):
        DAG = self.get_DAG(line, max_word_len)
        route = self.get_max_probability_route(DAG, line)
        return self.segment(line, route)