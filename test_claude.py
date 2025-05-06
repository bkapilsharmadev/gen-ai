# test_claude.py

from common.llm_chain import get_llm

def main():
    llm = get_llm()
    response = llm.invoke("Summarize this: Claude 3 is now available in AWS Bedrock.")
    print("✅ Claude response:\n")
    print(response)

if __name__ == "__main__":
    main()
