"""
PDF 추출 개선 테스트
"""

from improved_pdf_extractor import ImprovedPDFExtractor
import re

def analyze_pdf(pdf_path):
    """
    PDF 분석 및 문제점 진단
    """
    print(f"\n{'='*60}")
    print(f"PDF 분석: {pdf_path.split('/')[-1]}")
    print('='*60)
    
    # 개선된 추출기 사용
    extractor = ImprovedPDFExtractor(pdf_path)
    segments = extractor.extract_all(mode='smart')
    
    print(f"\n📊 추출 통계:")
    print(f"  • 총 세그먼트: {len(segments)}개")
    
    # 패턴별 분석
    patterns = {
        '① 형식': r'^[①②③④⑤⑥⑦⑧⑨⑩]',
        '❶ 형식': r'^[❶❷❸❹❺❻❼❽❾❿]',
        '1) 형식': r'^\d+\)',
        '가) 형식': r'^[가나다라마바사아자차]\)',
        '• 불릿': r'^[•▪▫◦]',
        '- 대시': r'^-\s',
    }
    
    pattern_counts = {}
    for name, pattern in patterns.items():
        matching = [s for s in segments if re.search(pattern, s)]
        pattern_counts[name] = len(matching)
        if matching:
            print(f"  • {name}: {len(matching)}개")
    
    # 길이 분포
    short = len([s for s in segments if len(s) < 50])
    medium = len([s for s in segments if 50 <= len(s) < 200])
    long = len([s for s in segments if len(s) >= 200])
    
    print(f"\n📏 길이 분포:")
    print(f"  • 짧음 (<50자): {short}개")
    print(f"  • 중간 (50-200자): {medium}개")
    print(f"  • 긴 (>200자): {long}개")
    
    # 문제가 있었던 부분 찾기
    print(f"\n🔍 주요 세그먼트 샘플:")
    
    # ②③ 패턴 찾기
    circle_segments = [s for s in segments if re.search(r'[②③]', s)]
    if circle_segments:
        print("\n[원 번호 항목]")
        for seg in circle_segments[:3]:
            preview = seg[:100] + "..." if len(seg) > 100 else seg
            print(f"  • {preview}")
    
    # ❶❷❸ 패턴 찾기
    black_circle = [s for s in segments if re.search(r'[❶❷❸]', s)]
    if black_circle:
        print("\n[검은원 번호 항목]")
        for seg in black_circle[:3]:
            preview = seg[:100] + "..." if len(seg) > 100 else seg
            print(f"  • {preview}")
    
    # 테이블 관련 텍스트 찾기
    table_keywords = ['서비스구분', '우대내용', '우대조건', '기본서비스', '추가서비스']
    table_segments = [s for s in segments if any(kw in s for kw in table_keywords)]
    if table_segments:
        print("\n[테이블 관련]")
        for seg in table_segments[:3]:
            preview = seg[:100] + "..." if len(seg) > 100 else seg
            print(f"  • {preview}")
    
    return segments


def compare_extraction_methods(pdf_path):
    """
    추출 방식 비교
    """
    print(f"\n{'='*60}")
    print("추출 방식 비교")
    print('='*60)
    
    # 1. Smart 모드
    extractor = ImprovedPDFExtractor(pdf_path)
    smart_segments = extractor.extract_all(mode='smart')
    
    # 2. Table aware 모드
    table_segments = extractor.extract_all(mode='table_aware')
    
    # 3. Text only 모드
    text_segments = extractor.extract_all(mode='text_only')
    
    print(f"\n📊 추출 결과 비교:")
    print(f"  • Smart 모드: {len(smart_segments)}개")
    print(f"  • Table aware 모드: {len(table_segments)}개")
    print(f"  • Text only 모드: {len(text_segments)}개")
    
    # 각 모드의 샘플 출력
    print(f"\n[Smart 모드 샘플 - 처음 5개]")
    for i, seg in enumerate(smart_segments[:5], 1):
        preview = seg[:80] + "..." if len(seg) > 80 else seg
        print(f"  {i}. {preview}")


def main():
    """
    메인 테스트
    """
    # 테스트할 PDF 파일들
    pdf_files = [
        '/Users/inter4259/Desktop/은행 상품 설명서/10000831_pi.pdf',
        '/Users/inter4259/Desktop/은행 상품 설명서/11. (표준서식)외화보통예금 상품설명서(개정전문)2021.09.24.pdf',
    ]
    
    # 첫 번째 PDF 상세 분석
    if pdf_files:
        segments = analyze_pdf(pdf_files[0])
        
        # 추출 방식 비교 (선택적)
        # compare_extraction_methods(pdf_files[0])
        
        # 문제가 있었던 특정 텍스트 찾기
        print(f"\n{'='*60}")
        print("특정 문제 텍스트 확인")
        print('='*60)
        
        # "② 적용기준" 부분 찾기
        target_text = "② 적용기준"
        found = [s for s in segments if target_text in s]
        if found:
            print(f"\n✅ '{target_text}' 찾음:")
            for seg in found:
                print(f"  길이: {len(seg)}자")
                print(f"  내용: {seg[:200]}...")
        else:
            print(f"\n❌ '{target_text}'를 찾을 수 없음")
        
        # "❶ NH All100플랜적금" 찾기
        target_text = "❶ NH All100플랜적금"
        found = [s for s in segments if target_text in s]
        if found:
            print(f"\n✅ '{target_text}' 찾음:")
            for seg in found:
                print(f"  길이: {len(seg)}자")
                print(f"  내용: {seg[:200]}...")


if __name__ == "__main__":
    main()