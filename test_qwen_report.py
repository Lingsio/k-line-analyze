
import asyncio
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from dotenv import load_dotenv

# Ensure backend directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")
sys.path.append(backend_dir)

load_dotenv()

async def test_report():
    print("Initializing AutoAnalyzer...")
    try:
        from app.services.auto_analyzer import AutoAnalyzer
        analyzer = AutoAnalyzer()
        
        symbol = "AAPL"
        market = "us"
        
        print(f"Analyzing {symbol} ({market}) with LLM enabled...")
        
        # Force refresh to ensure we trigger the LLM call and image generation
        result = await analyzer.analyze_stock(
            symbol=symbol,
            market=market,
            use_llm=True,
            force_refresh=True
        )
        
        print("\nAnalysis Complete!")
        print("-" * 50)
        print(f"Overall Signal: {result.overall_signal}")
        print(f"Confidence: {result.overall_confidence}")
        
        if result.llm_analysis:
            print("\nLLM Report Content (First 500 chars):")
            print(result.llm_analysis[:500] + "...")
            
            # Save full report
            output_file = "qwen_report_test.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.llm_analysis)
            print(f"\nFull report saved to {output_file}")
        else:
            print("\nNO LLM ANALYSIS GENERATED.")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_report())
