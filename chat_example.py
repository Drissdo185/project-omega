"""
Simple example script demonstrating the chat functionality.
Run this to test the Q&A system without the UI.
"""

import asyncio
from app.chat.chat_service import ChatService


async def main():
    """Run example chat session"""
    
    print("=" * 60)
    print("Vision RAG Chat System - Example")
    print("=" * 60)
    
    # Initialize service
    print("\n1️⃣ Initializing chat service...")
    service = ChatService()
    
    # List available documents
    print("\n2️⃣ Available documents:")
    documents = service.list_documents()
    
    if not documents:
        print("❌ No documents found!")
        print("Please process a document first:")
        print("  - Run the Streamlit UI: streamlit run main.py")
        print("  - Or process a PDF programmatically")
        return
    
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc['name']} ({doc['page_count']} pages)")
        print(f"      ID: {doc['id']}")
        print(f"      Has summaries: {'✅' if doc['has_summaries'] else '❌'}")
    
    # Use the first document
    doc = documents[0]
    doc_id = doc['id']
    
    print(f"\n3️⃣ Using document: {doc['name']}")
    
    # Example questions
    questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Summarize the key information."
    ]
    
    print(f"\n4️⃣ Asking {len(questions)} questions...")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 60)
        
        # Ask the question
        result = await service.ask(
            document_id=doc_id,
            question=question,
            max_pages=5
        )
        
        # Display result
        print(f"\n💡 Answer:")
        print(f"   {result['answer']}")
        
        print(f"\n📊 Metadata:")
        print(f"   • Pages used: {result['page_numbers']}")
        print(f"   • Selection cost: ${result['selection_cost']:.4f}")
        print(f"   • Analysis cost: ${result['analysis_cost']:.4f}")
        print(f"   • Total cost: ${result['total_cost']:.4f}")
        
        print("=" * 60)
    
    # Show total session cost
    total_cost = service.get_total_cost()
    print(f"\n💰 Total session cost: ${total_cost:.4f}")
    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())

