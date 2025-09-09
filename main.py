#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from typing import List, Dict
import json
from datetime import datetime

from pdf_extractor import PDFExtractor
from sentence_analyzer import SentenceAnalyzer
from vocabulary_analyzer import VocabularyAnalyzer
from visualizer import ResultVisualizer

class TextDifficultyAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_extractor = PDFExtractor(pdf_path)
        self.sentence_analyzer = SentenceAnalyzer()
        self.vocabulary_analyzer = VocabularyAnalyzer()
        self.visualizer = ResultVisualizer()
        
    def analyze(self) -> List[Dict]:
        """PDF 문서의 문장별 난이도 분석"""
        print(f"PDF 파일 분석 시작: {self.pdf_path}")
        print("-" * 50)
        
        # 1. PDF에서 텍스트 추출
        print("1. PDF 텍스트 추출 중...")
        text = self.pdf_extractor.extract_text()
        if not text:
            print("텍스트 추출 실패!")
            return []
        
        # 2. 문장 분리
        print("2. 문장 분리 중...")
        sentences = self.pdf_extractor.get_sentences()
        print(f"   총 {len(sentences)}개 문장 추출")
        
        # 3. 각 문장 분석
        print("3. 문장별 난이도 분석 중...")
        results = []
        
        for idx, sentence in enumerate(sentences, 1):
            if idx % 10 == 0:
                print(f"   {idx}/{len(sentences)} 문장 분석 완료...")
            
            # 문장 수준 분석
            sentence_metrics = self.sentence_analyzer.analyze_sentence_difficulty(sentence)
            
            # 어휘 수준 분석
            vocabulary_metrics = self.vocabulary_analyzer.analyze_vocabulary_difficulty(sentence)
            
            # 종합 난이도 계산 (문장 난이도와 어휘 난이도의 가중 평균)
            total_difficulty = (
                sentence_metrics['difficulty_score'] * 0.5 + 
                vocabulary_metrics['vocab_difficulty_score'] * 0.5
            )
            
            results.append({
                'sentence_id': idx,
                'sentence': sentence,
                'sentence_metrics': sentence_metrics,
                'vocabulary_metrics': vocabulary_metrics,
                'total_difficulty': round(total_difficulty, 2)
            })
        
        print(f"분석 완료! 총 {len(results)}개 문장 분석됨")
        return results
    
    def print_summary(self, results: List[Dict]):
        """분석 결과 요약 출력"""
        if not results:
            print("분석 결과가 없습니다.")
            return
        
        print("\n" + "=" * 60)
        print("📊 분석 결과 요약")
        print("=" * 60)
        
        # 전체 통계
        total_sentences = len(results)
        avg_sentence_length = sum(r['sentence_metrics']['sentence_length'] for r in results) / total_sentences
        avg_total_difficulty = sum(r['total_difficulty'] for r in results) / total_sentences
        complex_sentences = sum(1 for r in results if r['sentence_metrics']['is_complex'])
        complex_ratio = (complex_sentences / total_sentences) * 100
        
        print(f"\n📝 전체 통계:")
        print(f"  • 총 문장 수: {total_sentences}개")
        print(f"  • 평균 문장 길이: {avg_sentence_length:.1f} 어절")
        print(f"  • 복문 비율: {complex_ratio:.1f}%")
        print(f"  • 평균 종합 난이도: {avg_total_difficulty:.2f}/10")
        
        # 난이도 분포
        difficulty_levels = {
            '매우 쉬움 (0-2)': 0,
            '쉬움 (2-4)': 0,
            '보통 (4-6)': 0,
            '어려움 (6-8)': 0,
            '매우 어려움 (8-10)': 0
        }
        
        for r in results:
            score = r['total_difficulty']
            if score <= 2:
                difficulty_levels['매우 쉬움 (0-2)'] += 1
            elif score <= 4:
                difficulty_levels['쉬움 (2-4)'] += 1
            elif score <= 6:
                difficulty_levels['보통 (4-6)'] += 1
            elif score <= 8:
                difficulty_levels['어려움 (6-8)'] += 1
            else:
                difficulty_levels['매우 어려움 (8-10)'] += 1
        
        print(f"\n📈 난이도 분포:")
        for level, count in difficulty_levels.items():
            percentage = (count / total_sentences) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {level:15s}: {bar:25s} {count:3d}개 ({percentage:5.1f}%)")
        
        # 가장 어려운 문장 Top 5
        top_difficult = sorted(results, key=lambda x: x['total_difficulty'], reverse=True)[:5]
        
        print(f"\n🔝 가장 어려운 문장 Top 5:")
        for i, r in enumerate(top_difficult, 1):
            sentence_preview = r['sentence'][:50] + '...' if len(r['sentence']) > 50 else r['sentence']
            print(f"  {i}. [난이도 {r['total_difficulty']:.2f}] {sentence_preview}")
        
        # 가장 쉬운 문장 Top 5
        top_easy = sorted(results, key=lambda x: x['total_difficulty'])[:5]
        
        print(f"\n✅ 가장 쉬운 문장 Top 5:")
        for i, r in enumerate(top_easy, 1):
            sentence_preview = r['sentence'][:50] + '...' if len(r['sentence']) > 50 else r['sentence']
            print(f"  {i}. [난이도 {r['total_difficulty']:.2f}] {sentence_preview}")
    
    def save_results(self, results: List[Dict], output_dir: str = None):
        """분석 결과 저장"""
        if not output_dir:
            output_dir = os.path.dirname(self.pdf_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        
        # JSON 저장
        json_path = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 JSON 결과 저장: {json_path}")
        
        # Excel 보고서 생성
        excel_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.xlsx")
        self.visualizer.create_detailed_report(results, excel_path)
        
        # 시각화
        print("\n📊 시각화 생성 중...")
        df = self.visualizer.create_summary_dataframe(results)
        
        # 그래프 저장
        plot_path = os.path.join(output_dir, f"{base_name}_plots_{timestamp}.png")
        self.visualizer.plot_difficulty_distribution(df, save_path=plot_path)
        
        # Top 10 어려운 문장 그래프
        self.visualizer.plot_top_difficult_sentences(df, top_n=10)
        
        return json_path, excel_path

def main():
    parser = argparse.ArgumentParser(description='PDF 문서의 문장별 난이도 분석')
    parser.add_argument('pdf_path', type=str, help='분석할 PDF 파일 경로')
    parser.add_argument('--output', '-o', type=str, help='결과 저장 디렉토리', default=None)
    parser.add_argument('--no-viz', action='store_true', help='시각화 생략')
    
    args = parser.parse_args()
    
    # PDF 파일 존재 확인
    if not os.path.exists(args.pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다: {args.pdf_path}")
        sys.exit(1)
    
    # 분석 실행
    analyzer = TextDifficultyAnalyzer(args.pdf_path)
    results = analyzer.analyze()
    
    if results:
        # 요약 출력
        analyzer.print_summary(results)
        
        # 결과 저장
        if not args.no_viz:
            analyzer.save_results(results, args.output)
    else:
        print("분석에 실패했습니다.")

if __name__ == "__main__":
    # 테스트용 직접 실행
    if len(sys.argv) == 1:
        # 기본 PDF 경로 사용
        test_pdf = "/Users/inter4259/Desktop/은행 상품 설명서/10000831_pi.pdf"
        if os.path.exists(test_pdf):
            print("테스트 모드: 기본 PDF 파일 분석")
            analyzer = TextDifficultyAnalyzer(test_pdf)
            results = analyzer.analyze()
            if results:
                analyzer.print_summary(results)
                analyzer.save_results(results)
        else:
            print("사용법: python main.py <PDF 파일 경로>")
    else:
        main()