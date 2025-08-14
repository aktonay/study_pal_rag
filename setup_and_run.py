#!/usr/bin/env python3
"""
Setup and run script for the Bilingual RAG Web Application
"""

import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "templates",
        "temp_uploads",
        "persistent_vector_store",
        "document_storage"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}/")
    
    print("✅ Directories created successfully!")

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created!")
        print("⚠️  Please add your OpenAI API key to the .env file")
        return False
    
    # Check if API key is set
    with open(env_file, "r") as f:
        content = f.read()
        if "your_openai_api_key_here" in content:
            print("⚠️  Please add your actual OpenAI API key to the .env file")
            return False
    
    print("✅ .env file configured!")
    return True

def open_browser():
    """Open browser after a delay"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Browser opened at http://localhost:5000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please manually open http://localhost:5000 in your browser")

def run_application():
    """Run the Flask application"""
    print("🚀 Starting Bilingual RAG Web Application...")
    print("📖 Upload PDF/DOCX files and ask questions in English or Bengali")
    print("🌐 The application will be available at: http://localhost:5000")
    print("⏳ Starting server...")
    
    # Open browser in background
    timer = Timer(3.0, open_browser)
    timer.start()
    
    try:
        # Run the Flask app
        subprocess.call([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main setup and run function"""
    print("🤖 Bilingual RAG System Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed: Could not install requirements")
        return
    
    # Check environment file
    if not check_env_file():
        print("\n❌ Setup incomplete:")
        print("1. Edit the .env file")
        print("2. Add your OpenAI API key")
        print("3. Run this script again")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nApplication Features:")
    print("✅ Persistent document storage (no re-upload needed)")
    print("✅ Bilingual support (English + Bengali)")
    print("✅ Short-term memory (chat history)")
    print("✅ Long-term memory (document corpus)")
    print("✅ Web interface with drag & drop upload")
    print("✅ Language-aware responses")
    
    # Run the application
    print("\n" + "=" * 40)
    run_application()

if __name__ == "__main__":
    main()