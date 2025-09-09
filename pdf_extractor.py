import pdfplumber
import re
from typing import List, Dict

class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text = ""
        self.pages = []
    
    def extract_text(self) -> str:
        """PDF에서 텍스트 추출"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        self.pages.append({
                            'page_num': page_num,
                            'text': page_text
                        })
                        self.text += page_text + "\n"
            return self.text
        except Exception as e:
            print(f"PDF 추출 오류: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수 기호 정리
        text = re.sub(r'[◆▪■●※]', '', text)
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def get_sentences(self) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        if not self.text:
            self.extract_text()
        
        # 문장 분리 규칙 개선
        # 1. 마침표, 느낌표, 물음표로 끝나는 경우
        # 2. 단, 숫자나 영문 약어 뒤의 마침표는 제외
        sentences = []
        
        # 기본 문장 분리
        text = self.clean_text(self.text)
        
        # 문장 종결 패턴
        sentence_endings = re.compile(r'([.!?])\s+(?=[가-힣A-Z])')
        raw_sentences = sentence_endings.split(text)
        
        # 분리된 문장 재조합
        current_sentence = ""
        for i, part in enumerate(raw_sentences):
            if part in '.!?':
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # 마지막 문장 처리
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 너무 짧은 문장 필터링 (3어절 미만)
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences