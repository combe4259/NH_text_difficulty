"""
개선된 PDF 텍스트 추출기
금융/법률 문서의 복잡한 구조를 정확하게 추출
"""

import pdfplumber
import re
from typing import List, Dict, Tuple
import pandas as pd


class ImprovedPDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pages = []
        self.tables = []
        self.text_segments = []
        
    def extract_all(self, mode='smart') -> List[str]:
        """
        PDF에서 모든 텍스트를 구조화하여 추출
        
        mode:
            - 'smart': 테이블과 텍스트를 지능적으로 분리
            - 'text_only': 텍스트만 추출
            - 'table_aware': 테이블을 우선 처리
        """
        segments = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            print(f"📄 PDF 분석 중: {self.pdf_path}")
            print(f"   총 {len(pdf.pages)} 페이지")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"   페이지 {page_num} 처리 중...")
                
                if mode == 'smart':
                    page_segments = self._extract_smart_page(page)
                elif mode == 'table_aware':
                    page_segments = self._extract_with_tables(page)
                else:
                    page_segments = self._extract_text_only(page)
                
                segments.extend(page_segments)
        
        # 후처리: 중복 제거 및 정리
        unique_segments = self._post_process_segments(segments)
        
        print(f"✅ 총 {len(unique_segments)}개 세그먼트 추출 완료")
        return unique_segments
    
    def _extract_smart_page(self, page) -> List[str]:
        """
        페이지를 스마트하게 분석하여 추출
        """
        segments = []
        
        # 1. 테이블 영역 찾기
        tables = page.extract_tables()
        table_bboxes = []
        
        if tables:
            # 테이블 영역의 좌표 저장
            for table in page.find_tables():
                table_bboxes.append(table.bbox)
                
                # 테이블 내용을 구조화하여 추출
                table_segments = self._process_table(table.extract())
                segments.extend(table_segments)
        
        # 2. 테이블 외 텍스트 추출
        # 테이블 영역을 제외한 텍스트만 추출
        text = page.extract_text()
        if text:
            # 테이블 영역의 텍스트 제거 (간단한 방법)
            text_segments = self._extract_non_table_text(text, tables)
            segments.extend(text_segments)
        
        return segments
    
    def _extract_with_tables(self, page) -> List[str]:
        """
        테이블을 우선적으로 처리하여 추출
        """
        segments = []
        
        # 테이블 추출
        tables = page.extract_tables()
        
        if tables:
            for table in tables:
                table_segments = self._process_table(table)
                segments.extend(table_segments)
        
        # 일반 텍스트도 추출
        text = page.extract_text()
        if text:
            text_segments = self._split_by_patterns(text)
            segments.extend(text_segments)
        
        return segments
    
    def _extract_text_only(self, page) -> List[str]:
        """
        텍스트만 추출 (기존 방식)
        """
        text = page.extract_text()
        if text:
            return self._split_by_patterns(text)
        return []
    
    def _process_table(self, table: List[List]) -> List[str]:
        """
        테이블 데이터를 의미있는 세그먼트로 변환
        """
        segments = []
        
        if not table or len(table) == 0:
            return segments
        
        # 테이블 헤더가 있는 경우
        has_header = self._detect_table_header(table[0])
        
        if has_header and len(table) > 1:
            headers = table[0]
            
            for row in table[1:]:
                if row and any(cell for cell in row if cell):
                    # 각 행을 독립적인 세그먼트로
                    row_text = self._format_table_row(headers, row)
                    if row_text and len(row_text) > 10:
                        segments.append(row_text)
        else:
            # 헤더가 없는 경우 각 행을 그대로 처리
            for row in table:
                if row and any(cell for cell in row if cell):
                    row_text = ' '.join([str(cell) if cell else '' for cell in row])
                    row_text = row_text.strip()
                    if len(row_text) > 10:
                        segments.append(row_text)
        
        return segments
    
    def _detect_table_header(self, row: List) -> bool:
        """
        테이블 헤더인지 감지
        """
        if not row:
            return False
        
        header_keywords = [
            '구분', '내용', '조건', '항목', '서비스구분', '우대내용',
            '우대조건', '적용기준', '비고', '설명', '종류', '금액'
        ]
        
        row_text = ' '.join([str(cell) if cell else '' for cell in row])
        
        for keyword in header_keywords:
            if keyword in row_text:
                return True
        
        return False
    
    def _format_table_row(self, headers: List, row: List) -> str:
        """
        테이블 행을 의미있는 텍스트로 포맷
        """
        formatted = []
        
        for i, cell in enumerate(row):
            if cell and str(cell).strip():
                if i < len(headers) and headers[i]:
                    # 헤더가 있으면 "헤더: 내용" 형식
                    formatted.append(f"{headers[i]}: {cell}")
                else:
                    formatted.append(str(cell))
        
        return ' | '.join(formatted)
    
    def _extract_non_table_text(self, text: str, tables: List) -> List[str]:
        """
        테이블이 아닌 텍스트만 추출
        """
        # 간단한 휴리스틱: 테이블에 포함된 텍스트 제거
        # (실제로는 bbox 좌표를 사용해야 더 정확함)
        
        segments = []
        
        # 테이블 내용을 텍스트에서 제거
        for table in tables:
            if table:
                for row in table:
                    row_text = ' '.join([str(cell) if cell else '' for cell in row])
                    text = text.replace(row_text, '')
        
        # 남은 텍스트를 패턴별로 분리
        return self._split_by_patterns(text)
    
    def _split_by_patterns(self, text: str) -> List[str]:
        """
        패턴 기반 텍스트 분리 (개선된 버전)
        """
        segments = []
        
        # 줄 단위로 처리
        lines = text.split('\n')
        current_segment = []
        current_marker = None
        
        # 패턴 정의
        patterns = {
            'main_num': r'^[①②③④⑤⑥⑦⑧⑨⑩]',
            'sub_num': r'^[❶❷❸❹❺❻❼❽❾❿]',
            'paren_num': r'^[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]',
            'number': r'^\d+[.)]\s',
            'korean': r'^[가나다라마바사아자차][.)]\s',
            'bullet': r'^[-•▪▫◦※*]\s',
            'article': r'^제\s*\d+\s*[조항관]',
        }
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # 빈 줄에서 세그먼트 종료
                if current_segment:
                    segment_text = ' '.join(current_segment).strip()
                    if len(segment_text) > 10:
                        segments.append(segment_text)
                    current_segment = []
                    current_marker = None
                continue
            
            # 패턴 매칭
            matched = False
            for pattern_name, pattern in patterns.items():
                if re.match(pattern, line):
                    # 이전 세그먼트 저장
                    if current_segment:
                        segment_text = ' '.join(current_segment).strip()
                        if len(segment_text) > 10:
                            segments.append(segment_text)
                    
                    # 새 세그먼트 시작
                    current_segment = [line]
                    current_marker = pattern_name
                    matched = True
                    break
            
            if not matched:
                if current_segment:
                    # 현재 세그먼트에 추가
                    current_segment.append(line)
                elif len(line) > 10:
                    # 독립 세그먼트
                    segments.append(line)
        
        # 마지막 세그먼트 저장
        if current_segment:
            segment_text = ' '.join(current_segment).strip()
            if len(segment_text) > 10:
                segments.append(segment_text)
        
        return segments
    
    def _post_process_segments(self, segments: List[str]) -> List[str]:
        """
        세그먼트 후처리
        """
        processed = []
        seen = set()
        
        for segment in segments:
            # 정규화
            normalized = ' '.join(segment.split())
            
            # 너무 긴 세그먼트는 분리
            if len(normalized) > 500:
                # 문장 단위로 분리
                sentences = re.split(r'(?<=[.!?])\s+', normalized)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 10 and sent not in seen:
                        seen.add(sent)
                        processed.append(sent)
            elif len(normalized) > 10 and normalized not in seen:
                seen.add(normalized)
                processed.append(normalized)
        
        return processed


def test_extractor():
    """
    테스트 함수
    """
    pdf_path = '/Users/inter4259/Desktop/은행 상품 설명서/10000831_pi.pdf'
    
    extractor = ImprovedPDFExtractor(pdf_path)
    segments = extractor.extract_all(mode='smart')
    
    print("\n=== 추출 결과 샘플 ===")
    for i, segment in enumerate(segments[:20], 1):
        preview = segment[:80] + "..." if len(segment) > 80 else segment
        print(f"{i:2d}. {preview}")
    
    # 패턴별 통계
    patterns = {
        '①②③': r'[①②③④⑤⑥⑦⑧⑨⑩]',
        '❶❷❸': r'[❶❷❸❹❺❻❼❽❾❿]',
        '1)2)3)': r'^\d+\)',
    }
    
    print("\n=== 패턴별 세그먼트 수 ===")
    for name, pattern in patterns.items():
        count = sum(1 for s in segments if re.search(pattern, s))
        print(f"{name}: {count}개")
    
    return segments


if __name__ == "__main__":
    test_extractor()