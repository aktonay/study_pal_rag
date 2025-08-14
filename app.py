from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from dotenv import load_dotenv
from rag_pipeline_components import EnhancedRAGPipeline  # ✅ Corrected import
import json
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Initialize RAG Pipeline
rag_pipeline = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    try:
        rag_pipeline = EnhancedRAGPipeline(
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        print("✅ RAG Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing RAG Pipeline: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_pipeline is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            try:
                # Secure filename
                filename = secure_filename(file.filename)

                # Save temporary file
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)

                # Process document
                result = rag_pipeline.process_document(temp_path, filename)

                # Clean up temporary file
                os.remove(temp_path)

                results.append({
                    'filename': filename,
                    'success': result['success'],
                    'message': result['message'],
                    'chunks_added': result['chunks_added'],
                    'already_exists': result.get('already_exists', False),
                    'language': result.get('language', 'unknown')
                })

            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'message': f'Error processing file: {str(e)}',
                    'chunks_added': 0
                })
        else:
            results.append({
                'filename': file.filename,
                'success': False,
                'message': 'Invalid file type. Only PDF and DOCX files are allowed.',
                'chunks_added': 0
            })

    return jsonify({'results': results})

@app.route('/api/query', methods=['POST'])
def query():
    """Handle chat queries"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query'].strip()
    if not user_query:
        return jsonify({'error': 'Empty query'}), 400

    try:
        # Generate answer
        result = rag_pipeline.generate_answer(user_query)

        return jsonify({
            'answer': result['answer'],
            'query_language': result['query_language'],
            'sources_count': result['sources_count'],
            'sources': result['retrieved_documents'],
            'success': result['success']
        })

    except Exception as e:
        return jsonify({
            'error': f'Error processing query: {str(e)}',
            'success': False
        }), 500

@app.route('/api/documents')
def get_documents():
    """Get list of processed documents"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    try:
        documents = rag_pipeline.get_processed_documents()
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    try:
        stats = rag_pipeline.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search documents without generating answer"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    try:
        results = rag_pipeline.search_documents(data['query'], k=data.get('k', 10))
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    try:
        rag_pipeline.clear_chat_history()
        return jsonify({'success': True, 'message': 'Chat history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-kb', methods=['POST'])
def reset_knowledge_base():
    """Reset knowledge base"""
    if rag_pipeline is None:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500

    try:
        rag_pipeline.reset_knowledge_base()
        return jsonify({'success': True, 'message': 'Knowledge base reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html')

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file")
        exit(1)

    # Initialize RAG Pipeline
    if not initialize_rag_pipeline():
        print("❌ Failed to initialize RAG Pipeline")
        exit(1)

    # Run the app
    print("🚀 Starting Bilingual RAG Web Application")
    print("📖 Upload PDF/DOCX files and ask questions in English or Bengali")
    print("🌐 Access the application at: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
