import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import matplotlib.font_manager as fm
import numpy as np
import platform
import os

class ResultVisualizer:
    def __init__(self):
        # 한글 폰트 설정
        self.font_prop = self.setup_korean_font()
        
    def setup_korean_font(self):
        """한글 폰트 설정"""
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # 직접 폰트 경로 사용
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
            
            if os.path.exists(font_path):
                from matplotlib import font_manager, rc
                
                # 폰트 프로퍼티 생성
                font_prop = font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                
                # 폰트 매니저에 등록
                font_manager.fontManager.addfont(font_path)
                
                # 모든 폰트 설정을 강제로 변경
                rc('font', family=font_name)
                rc('axes', unicode_minus=False)
                
                # 추가 설정
                plt.rcParams['font.family'] = font_name
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                
                # 백엔드 설정
                import matplotlib
                matplotlib.use('Agg')  # 비대화형 백엔드 사용
                
                # seaborn 설정
                sns.set(font=font_name, rc={'axes.unicode_minus': False})
                
                print(f"한글 폰트 설정 완료: {font_name}")
                return font_prop
        
        # Windows
        elif system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['font.sans-serif'] = ['Malgun Gothic']
        else:  # Linux
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['font.sans-serif'] = ['NanumGothic']
        
        plt.rcParams['axes.unicode_minus'] = False
        return None
    
    def create_summary_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """분석 결과를 DataFrame으로 변환"""
        summary_data = []
        
        for idx, result in enumerate(results, 1):
            summary_data.append({
                '문장번호': idx,
                '문장': result['sentence'][:50] + '...' if len(result['sentence']) > 50 else result['sentence'],
                '문장길이': result['sentence_metrics']['sentence_length'],
                '구조점수': result['sentence_metrics']['structure_score'],
                '복문여부': '복문' if result['sentence_metrics']['is_complex'] else '단문',
                '명사/서술어비율': result['sentence_metrics']['noun_predicate_ratio'],
                '어휘다양성': result['vocabulary_metrics']['vocabulary_diversity'],
                '어려운단어비율': result['vocabulary_metrics']['difficult_word_ratio'],
                '문장난이도': result['sentence_metrics']['difficulty_score'],
                '어휘난이도': result['vocabulary_metrics']['vocab_difficulty_score'],
                '종합난이도': result['total_difficulty']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_difficulty_distribution(self, df: pd.DataFrame, save_path: str = None):
        """난이도 분포 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 종합 난이도 분포
        axes[0, 0].hist(df['종합난이도'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('종합 난이도 분포', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('난이도 점수')
        axes[0, 0].set_ylabel('문장 수')
        axes[0, 0].axvline(df['종합난이도'].mean(), color='red', linestyle='--', label=f'평균: {df["종합난이도"].mean():.2f}')
        axes[0, 0].legend()
        
        # 2. 문장 길이 vs 난이도
        axes[0, 1].scatter(df['문장길이'], df['종합난이도'], alpha=0.6, color='darkgreen')
        axes[0, 1].set_title('문장 길이와 난이도 관계', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('문장 길이 (어절 수)')
        axes[0, 1].set_ylabel('종합 난이도')
        
        # 추세선 추가
        z = np.polyfit(df['문장길이'], df['종합난이도'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df['문장길이'], p(df['문장길이']), "r--", alpha=0.5)
        
        # 3. 문장 유형별 난이도
        complex_data = df.groupby('복문여부')['종합난이도'].mean()
        axes[1, 0].bar(complex_data.index, complex_data.values, color=['skyblue', 'coral'])
        axes[1, 0].set_title('문장 유형별 평균 난이도', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('평균 난이도')
        
        # 4. 어휘 vs 문장 난이도
        axes[1, 1].scatter(df['문장난이도'], df['어휘난이도'], alpha=0.6, color='purple')
        axes[1, 1].set_title('문장 난이도 vs 어휘 난이도', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('문장 난이도')
        axes[1, 1].set_ylabel('어휘 난이도')
        axes[1, 1].plot([0, 10], [0, 10], 'k--', alpha=0.3)  # 대각선
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프 저장 완료: {save_path}")
        plt.close()  # show() 대신 close() 사용
    
    def plot_top_difficult_sentences(self, df: pd.DataFrame, top_n: int = 10):
        """가장 어려운 문장 Top N 시각화"""
        top_sentences = df.nlargest(top_n, '종합난이도')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(top_sentences))
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_sentences)))
        
        bars = ax.barh(y_pos, top_sentences['종합난이도'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"문장 {idx}" for idx in top_sentences['문장번호']])
        ax.invert_yaxis()
        ax.set_xlabel('종합 난이도 점수')
        ax.set_title(f'가장 어려운 문장 Top {top_n}', fontsize=16, fontweight='bold')
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, top_sentences['종합난이도'])):
            ax.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center')
        
        plt.tight_layout()
        plt.savefig('top_difficult_sentences.png', dpi=300, bbox_inches='tight')
        print("Top 어려운 문장 그래프 저장: top_difficult_sentences.png")
        plt.close()
    
    def create_detailed_report(self, results: List[Dict], output_path: str):
        """상세 분석 보고서 생성 (Excel)"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 요약 시트
            df_summary = self.create_summary_dataframe(results)
            df_summary.to_excel(writer, sheet_name='요약', index=False)
            
            # 2. 상세 분석 시트
            detailed_data = []
            for idx, result in enumerate(results, 1):
                detailed_data.append({
                    '문장번호': idx,
                    '문장': result['sentence'],
                    '문장길이': result['sentence_metrics']['sentence_length'],
                    '구조점수': result['sentence_metrics']['structure_score'],
                    '복문여부': result['sentence_metrics']['is_complex'],
                    '명사/서술어비율': result['sentence_metrics']['noun_predicate_ratio'],
                    '어휘다양성': result['vocabulary_metrics']['vocabulary_diversity'],
                    '어려운단어비율': result['vocabulary_metrics']['difficult_word_ratio'],
                    '한자어비율': result['vocabulary_metrics']['sino_korean_ratio'],
                    '금융용어': ', '.join(result['vocabulary_metrics']['difficult_words']['financial']),
                    '법률용어': ', '.join(result['vocabulary_metrics']['difficult_words']['legal']),
                    '문장난이도': result['sentence_metrics']['difficulty_score'],
                    '어휘난이도': result['vocabulary_metrics']['vocab_difficulty_score'],
                    '종합난이도': result['total_difficulty']
                })
            
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_excel(writer, sheet_name='상세분석', index=False)
            
            # 3. 통계 시트
            stats_data = {
                '지표': ['평균 문장 길이', '평균 구조 점수', '복문 비율', 
                        '평균 어휘 다양성', '평균 어려운 단어 비율',
                        '평균 문장 난이도', '평균 어휘 난이도', '평균 종합 난이도'],
                '값': [
                    df_summary['문장길이'].mean(),
                    df_summary['구조점수'].mean(),
                    (df_summary['복문여부'] == '복문').mean() * 100,
                    df_summary['어휘다양성'].mean(),
                    df_summary['어려운단어비율'].mean(),
                    df_summary['문장난이도'].mean(),
                    df_summary['어휘난이도'].mean(),
                    df_summary['종합난이도'].mean()
                ]
            }
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='통계', index=False)
        
        print(f"상세 보고서가 {output_path}에 저장되었습니다.")