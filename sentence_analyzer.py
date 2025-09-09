import re
from kiwipiepy import Kiwi
from typing import List, Dict, Tuple
from collections import Counter

class SentenceAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        
    def tokenize(self, sentence: str) -> Dict:
        """문장을 형태소 분석하여 토큰화"""
        # 형태소 분석
        result = self.kiwi.tokenize(sentence)
        pos_tags = [(token.form, token.tag) for token in result]
        
        # 품사별 분류
        tokens = {
            'nouns': [],
            'verbs': [],
            'adjectives': [],
            'josa': [],  # 조사
            'eomi': [],  # 어미
            'all_pos': pos_tags
        }
        
        for word, pos in pos_tags:
            if pos.startswith('N'):  # NNG, NNP, NNB 등 명사류
                tokens['nouns'].append(word)
            elif pos.startswith('V'):  # VV, VA 등 동사류
                tokens['verbs'].append(word)
            elif pos == 'VA':  # 형용사
                tokens['adjectives'].append(word)
            elif pos.startswith('J'):  # JKS, JKO 등 조사류
                tokens['josa'].append(word)
            elif pos.startswith('E'):  # EP, EF, EC 등 어미류
                tokens['eomi'].append(word)
                
        return tokens
    
    def calculate_sentence_length(self, sentence: str) -> int:
        """문장의 어절 수 계산"""
        # 공백으로 분리하여 어절 수 계산
        words = sentence.split()
        return len(words)
    
    def calculate_structure_score(self, tokens: Dict) -> float:
        """문장 구조 복잡도 점수 계산"""
        score = 0
        
        # 조사 복잡도 (다양한 조사 사용)
        josa_variety = len(set(tokens['josa'])) if tokens['josa'] else 0
        score += josa_variety * 0.5
        
        # 어미 복잡도 (연결어미, 종결어미 등)
        eomi_count = len(tokens['eomi'])
        score += eomi_count * 0.3
        
        # 품사 다양성
        pos_variety = sum([
            1 for key in ['nouns', 'verbs', 'adjectives'] 
            if tokens[key]
        ])
        score += pos_variety * 0.2
        
        return score
    
    def detect_complex_sentence(self, sentence: str) -> bool:
        """복문 여부 판단"""
        # 복문 판단 패턴
        complex_patterns = [
            r'[^,]\s*,\s*[가-힣]+',  # 쉼표로 연결된 절
            r'(고|며|면서|지만|으나|거나|든지)',  # 연결어미
            r'(하여|해서|하고|이고|이며)',  # 연결 표현
            r'(때문에|위해|위하여|따라|의해)',  # 인과/목적 표현
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, sentence):
                return True
        return False
    
    def calculate_noun_predicate_ratio(self, tokens: Dict) -> float:
        """명사/서술어 비율 계산"""
        noun_count = len(tokens['nouns'])
        predicate_count = len(tokens['verbs']) + len(tokens['adjectives'])
        
        if predicate_count == 0:
            return float('inf') if noun_count > 0 else 0
        
        return noun_count / predicate_count
    
    def analyze_sentence_difficulty(self, sentence: str) -> Dict:
        """문장 난이도 종합 분석"""
        tokens = self.tokenize(sentence)
        
        # 문장 수준 지표 계산
        sentence_length = self.calculate_sentence_length(sentence)
        structure_score = self.calculate_structure_score(tokens)
        is_complex = self.detect_complex_sentence(sentence)
        noun_predicate_ratio = self.calculate_noun_predicate_ratio(tokens)
        
        # 난이도 점수 계산 (0-10 스케일)
        difficulty_score = 0
        
        # 문장 길이 기준 (10어절 이하: 0점, 30어절 이상: 3점)
        if sentence_length <= 10:
            difficulty_score += 0
        elif sentence_length <= 20:
            difficulty_score += 1.5
        elif sentence_length <= 30:
            difficulty_score += 2.5
        else:
            difficulty_score += 3
        
        # 구조 복잡도 (0-3점)
        difficulty_score += min(structure_score / 3, 3)
        
        # 복문 여부 (2점)
        if is_complex:
            difficulty_score += 2
        
        # 명사/서술어 비율 (비율이 높을수록 어려움, 0-2점)
        if noun_predicate_ratio > 2:
            difficulty_score += 2
        elif noun_predicate_ratio > 1:
            difficulty_score += 1
        
        # 최종 점수 정규화 (0-10)
        difficulty_score = min(difficulty_score, 10)
        
        return {
            'sentence': sentence,
            'sentence_length': sentence_length,
            'structure_score': round(structure_score, 2),
            'is_complex': is_complex,
            'complex_ratio': 1.0 if is_complex else 0.0,
            'noun_predicate_ratio': round(noun_predicate_ratio, 2),
            'difficulty_score': round(difficulty_score, 2),
            'tokens': tokens
        }