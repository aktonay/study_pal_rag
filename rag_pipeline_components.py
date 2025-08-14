import os
from typing import List, Dict, Optional
from openai import OpenAI
from document_processor import DocumentProcessor
from vector_store import PersistentFAISSVectorStore
import json
from datetime import datetime

class EnhancedRAGPipeline:
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize Enhanced RAG Pipeline with persistence
        """
        self.client = OpenAI()
        self.llm_model = llm_model
        
        # Initialize components with persistence
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Determine embedding dimension
        dimension = 3072 if "large" in embedding_model else 1536
        self.vector_store = PersistentFAISSVectorStore(
            embedding_model=embedding_model, 
            dimension=dimension,
            storage_path="persistent_vector_store"
        )
        
        # Chat history for short-term memory (in-memory, session-based)
        self.chat_history = []
        self.max_history = 15  # Keep last 15 exchanges
        
        # Chat history persistence
        self.chat_history_file = "chat_sessions.json"
        self.load_recent_chat_history()
    
    def load_recent_chat_history(self):
        """Load recent chat history from disk"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    all_sessions = json.load(f)
                    # Load the most recent session
                    if all_sessions and len(all_sessions) > 0:
                        latest_session = max(all_sessions, key=lambda x: x.get('timestamp', ''))
                        self.chat_history = latest_session.get('messages', [])[-self.max_history:]
        except Exception as e:
            print(f"Error loading chat history: {e}")
    
    def save_chat_history(self):
        """Save current chat history to disk"""
        try:
            # Load existing sessions
            all_sessions = []
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    all_sessions = json.load(f)
            
            # Add current session
            current_session = {
                'timestamp': datetime.now().isoformat(),
                'messages': self.chat_history
            }
            all_sessions.append(current_session)
            
            # Keep only last 5 sessions
            all_sessions = all_sessions[-5:]
            
            # Save back
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(all_sessions, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def process_document(self, file_path: str, filename: str = None) -> Dict:
        """Process a single document and add to knowledge base with enhanced feedback"""
        try:
            if filename is None:
                filename = os.path.basename(file_path)
            
            print(f"📄 Processing document: {filename}")
            
            # Check if document is already processed
            chunks = self.document_processor.process_document(file_path, force_reprocess=False)
            
            if not chunks:
                # Check if it was already processed or failed
                if self.document_processor.is_document_processed(file_path, filename):
                    return {
                        'success': False,
                        'message': f"Document '{filename}' was already processed",
                        'chunks_added': 0,
                        'already_exists': True,
                        'extraction_status': 'skipped'
                    }
                else:
                    return {
                        'success': False,
                        'message': f"No text could be extracted from '{filename}'. This could be due to:\n- Scanned images (requires OCR)\n- Encrypted/protected PDF\n- Corrupted file\n- Unsupported encoding",
                        'chunks_added': 0,
                        'already_exists': False,
                        'extraction_status': 'failed'
                    }
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks)
            
            if success:
                return {
                    'success': True,
                    'message': f"✅ Successfully processed '{filename}'",
                    'chunks_added': len(chunks),
                    'already_exists': False,
                    'language': chunks[0]['metadata'].get('primary_language', 'unknown'),
                    'extraction_status': 'success',
                    'file_size': chunks[0]['metadata'].get('file_size', 0),
                    'total_chars': chunks[0]['metadata'].get('total_chars', 0)
                }
            else:
                return {
                    'success': False,
                    'message': f"❌ Failed to add '{filename}' to vector store",
                    'chunks_added': 0,
                    'already_exists': False,
                    'extraction_status': 'vector_store_failed'
                }
                
        except Exception as e:
            print(f"❌ Error processing '{filename}': {str(e)}")
            return {
                'success': False,
                'message': f"Error processing '{filename}': {str(e)}",
                'chunks_added': 0,
                'already_exists': False,
                'extraction_status': 'error'
            }
    
    def get_processed_documents(self) -> List[Dict]:
        """Get list of all processed documents with enhanced info"""
        docs = self.document_processor.get_processed_documents_info()
        # Add vector store statistics for each document
        for doc in docs:
            doc['status'] = 'processed'
        return docs
    
    def detect_query_language(self, text: str) -> str:
        """Detect language of the query"""
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        english_chars = sum(1 for char in text if char.isalpha() and not ('\u0980' <= char <= '\u09FF'))
        total_chars = bengali_chars + english_chars
        
        if total_chars == 0:
            return "english"  # Default to English for non-alphabetic queries
        
        bengali_ratio = bengali_chars / total_chars
        
        if bengali_ratio > 0.3:
            return "bengali"
        elif bengali_ratio < 0.1:
            return "english"
        else:
            return "mixed"
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents with language awareness"""
        query_language = self.detect_query_language(query)
        
        # Search with language preference
        if query_language == "bengali":
            results = self.vector_store.search(query, k=k, language_filter="bengali")
            # If no Bengali results, search all languages
            if not results:
                results = self.vector_store.search(query, k=k)
        elif query_language == "english":
            results = self.vector_store.search(query, k=k, language_filter="english")
            # If no English results, search all languages
            if not results:
                results = self.vector_store.search(query, k=k)
        else:
            # Mixed or unknown language - search all
            results = self.vector_store.search(query, k=k)
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict], max_context_length: int = 3000) -> str:
        """Format retrieved documents into context string with length limit"""
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            filename = doc['metadata'].get('filename', 'Unknown')
            score = doc.get('score', 0)
            
            doc_context = f"[Source {i}: {filename} (Relevance: {score:.3f})]\n{doc['text']}\n"
            
            # Check if adding this document would exceed the limit
            if current_length + len(doc_context) > max_context_length and context_parts:
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        return "\n".join(context_parts)
    
    def generate_system_prompt(self, query_language: str) -> str:
        """Generate system prompt based on query language"""
        if query_language == "bengali":
            return """আপনি একজন সহায়ক সহকারী যিনি প্রদান করা নথির উপর ভিত্তি করে প্রশ্নের উত্তর দেন।

নির্দেশনা:
১. শুধুমাত্র প্রসঙ্গ নথিতে প্রদত্ত তথ্য ব্যবহার করে প্রশ্নের উত্তর দিন
২. উত্তর নথিতে না থাকলে স্পষ্টভাবে বলুন যে তথ্যটি উপলব্ধ নেই
৩. নির্ভুল এবং বিস্তারিত উত্তর প্রদান করুন
৪. তথ্যের উৎস উল্লেখ করুন
৫. বাংলায় প্রশ্ন করা হলে বাংলায় উত্তর দিন
৬. সহায়ক, নির্ভুল এবং বিনয়ী হন"""
        
        elif query_language == "mixed":
            return """You are a helpful assistant that answers questions based on provided documents. You can respond in both English and Bengali.

Instructions:
1. Answer using ONLY the information from the context documents
2. If information is not available, clearly state so
3. Provide accurate and detailed responses
4. Cite relevant sources
5. Respond in the same language as the query
6. Be helpful, accurate, and honest

আপনি একজন সহায়ক সহকারী। ইংরেজি এবং বাংলা উভয় ভাষায় উত্তর দিতে পারেন।"""
        
        else:  # English or default
            return """You are a helpful assistant that answers questions based on the provided context documents.

Instructions:
1. Answer questions using ONLY the information provided in the context documents
2. If the answer is not in the context, clearly state that the information is not available
3. Provide accurate and detailed responses
4. Cite the relevant sources when providing information
5. If the user asks in Bengali, respond in Bengali
6. If the user asks in English, respond in English
7. Be helpful, accurate, and honest about the limitations of the available information"""
    
    def add_to_chat_history(self, query: str, response: str, query_language: str):
        """Add exchange to chat history (short-term memory)"""
        self.chat_history.append({
            "query": query,
            "response": response,
            "query_language": query_language,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        
        # Save to disk periodically
        if len(self.chat_history) % 5 == 0:  # Save every 5 messages
            self.save_chat_history()
    
    def format_chat_history(self, query_language: str) -> str:
        """Format chat history for context"""
        if not self.chat_history:
            return ""
        
        # Get recent relevant history (last 3-5 exchanges)
        recent_history = self.chat_history[-5:]
        
        if query_language == "bengali":
            history_text = "পূর্ববর্তী কথোপকথনের প্রসঙ্গ:\n"
        else:
            history_text = "Previous conversation context:\n"
        
        for exchange in recent_history:
            if query_language == "bengali":
                history_text += f"প্রশ্ন: {exchange['query']}\nউত্তর: {exchange['response'][:200]}...\n\n"
            else:
                history_text += f"Q: {exchange['query']}\nA: {exchange['response'][:200]}...\n\n"
        
        return history_text
    
    def generate_answer(self, query: str, max_tokens: int = 4000) -> Dict:
        """Generate answer using RAG pipeline with language detection"""
        
        # Detect query language
        query_language = self.detect_query_language(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(query, k=5)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Format chat history
        chat_context = self.format_chat_history(query_language)
        
        # Create prompts
        system_prompt = self.generate_system_prompt(query_language)
        
        # Build user prompt
        if query_language == "bengali":
            user_prompt = f"""প্রসঙ্গ নথিসমূহ:
{context}

{chat_context}

ব্যবহারকারীর প্রশ্ন: {query}

উপরের প্রসঙ্গ নথির উপর ভিত্তি করে একটি বিস্তারিত উত্তর প্রদান করুন।"""
        else:
            user_prompt = f"""Context Documents:
{context}

{chat_context}

User Question: {query}

Please provide a comprehensive answer based on the context documents above."""
        
        try:
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add to chat history
            self.add_to_chat_history(query, answer, query_language)
            
            return {
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "query_language": query_language,
                "sources_count": len(retrieved_docs),
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            if query_language == "bengali":
                error_msg = f"উত্তর তৈরিতে ত্রুটি: {e}"
            
            return {
                "answer": error_msg,
                "retrieved_documents": retrieved_docs,
                "query_language": query_language,
                "sources_count": len(retrieved_docs),
                "success": False
            }
    
    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        vector_stats = self.vector_store.get_stats()
        doc_stats = self.document_processor.get_processed_documents_info()
        
        return {
            "vector_store": vector_stats,
            "processed_documents": len(doc_stats),
            "chat_history_length": len(self.chat_history),
            "llm_model": self.llm_model,
            "recent_documents": doc_stats[:5]  # Last 5 processed documents
        }
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
        self.save_chat_history()
        print("Chat history cleared")
    
    def reset_knowledge_base(self):
        """Reset the entire knowledge base"""
        self.vector_store.clear_all_data()
        self.document_processor.clear_document_registry()
        print("Knowledge base completely reset")
    
    def search_documents(self, query: str, k: int = 10) -> List[Dict]:
        """Search documents without generating an answer"""
        return self.retrieve_relevant_documents(query, k=k)
    
    def diagnose_document_processing(self, file_path: str) -> Dict:
        """Diagnostic method to help troubleshoot document processing issues"""
        filename = os.path.basename(file_path)
        diagnosis = {
            'filename': filename,
            'file_exists': os.path.exists(file_path),
            'file_size': 0,
            'file_extension': '',
            'extraction_methods': {},
            'recommendations': []
        }
        
        if not diagnosis['file_exists']:
            diagnosis['recommendations'].append("File does not exist at the specified path")
            return diagnosis
        
        diagnosis['file_size'] = os.path.getsize(file_path)
        diagnosis['file_extension'] = os.path.splitext(file_path)[1].lower()
        
        if diagnosis['file_extension'] == '.pdf':
            # Test all PDF extraction methods
            methods = [
                ('PyMuPDF', self.document_processor.extract_text_with_pymupdf),
                ('pdfplumber', self.document_processor.extract_text_with_pdfplumber),
                ('PyPDF2', self.document_processor.extract_text_with_pypdf2)
            ]
            
            # Add OCR method if available
            try:
                import pytesseract
                methods.append(('OCR', self.document_processor.extract_text_with_ocr))
            except ImportError:
                diagnosis['recommendations'].append("OCR not available - install with: pip install pytesseract pillow pdf2image")
            
            for method_name, method_func in methods:
                try:
                    text = method_func(file_path)
                    diagnosis['extraction_methods'][method_name] = {
                        'success': bool(text.strip()),
                        'text_length': len(text),
                        'first_100_chars': text[:100] if text else ""
                    }
                except Exception as e:
                    diagnosis['extraction_methods'][method_name] = {
                        'success': False,
                        'error': str(e),
                        'text_length': 0
                    }
            
            # Provide recommendations
            successful_methods = [name for name, result in diagnosis['extraction_methods'].items() 
                                 if result['success']]
            
            if not successful_methods:
                diagnosis['recommendations'].extend([
                    "No text extraction method succeeded",
                    "This PDF appears to contain scanned images instead of text",
                    "Install OCR dependencies: pip install pytesseract pillow pdf2image",
                    "For Windows: Also install Tesseract executable from https://github.com/UB-Mannheim/tesseract/wiki",
                    "Check if the PDF is password protected or corrupted"
                ])
            else:
                diagnosis['recommendations'].append(f"Text extraction successful with: {', '.join(successful_methods)}")
        
        elif diagnosis['file_extension'] == '.docx':
            try:
                text = self.document_processor.extract_text_from_docx(file_path)
                diagnosis['extraction_methods']['python-docx'] = {
                    'success': bool(text.strip()),
                    'text_length': len(text),
                    'first_100_chars': text[:100] if text else ""
                }
            except Exception as e:
                diagnosis['extraction_methods']['python-docx'] = {
                    'success': False,
                    'error': str(e),
                    'text_length': 0
                }
        
        else:
            diagnosis['recommendations'].append(f"Unsupported file format: {diagnosis['file_extension']}")
        
        return diagnosis