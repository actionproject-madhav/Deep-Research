from flask import Flask, render_template, request, jsonify
import os
import time
import threading
import uuid
from dotenv import load_dotenv
from research_agent import ResearchAgent
from report_generator import ReportGenerator

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey123")
app.config['TEMPLATES_AUTO_RELOAD'] = True

research_tasks = {}

class ResearchThread(threading.Thread):
    def __init__(self, task_id, question):
        super().__init__()
        self.task_id = task_id
        self.question = question
        self.research_agent = ResearchAgent(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            google_api_key=os.environ["GOOGLE_API_KEY"],
            search_engine_id=os.environ["GOOGLE_SEARCH_ENGINE_ID"]
        )
        self.report_generator = ReportGenerator(os.environ["OPENAI_API_KEY"])
        self.progress = {
            'current_step': 0,
            'total_steps': 4,
            'steps': [
                "Analyzing question",
                "Searching academic sources",
                "Synthesizing information",
                "Writing report"
            ]
        }

    def update_progress(self, step, message=None):
        self.progress['current_step'] = step
        if message:
            research_tasks[self.task_id]['messages'].append({
                'type': 'system',
                'content': message,
                'timestamp': time.time()
            })

    def run(self):
        try:
            research_tasks[self.task_id] = {
                'status': 'processing',
                'progress': self.progress,
                'messages': [],
                'report': None
            }

            # Step 1: Analyze question
            self.update_progress(1, "🔍 Analyzing question scope...")
            research_results = self.research_agent.research(
                question=self.question,
                max_depth=1,
                results_per_query=3
            )

            # Step 2: Search sources
            self.update_progress(2, "📚 Gathering academic sources...")
            time.sleep(0.5)

            # Step 3: Synthesize information
            self.update_progress(3, "🧠 Analyzing findings...")
            time.sleep(0.5)

            # Step 4: Generate report
            self.update_progress(4, "📝 Finalizing report...")
            report = self.report_generator.generate_report(research_results)
            
            research_tasks[self.task_id].update({
                'status': 'completed',
                'report': report,
                'completed_at': time.time()
            })

        except Exception as e:
            research_tasks[self.task_id] = {
                'status': 'error',
                'error': str(e),
                'stack_trace': repr(e)
            }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/research', methods=['POST'])
def create_research_task():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a research question'}), 400
    
    task_id = str(uuid.uuid4())
    research_tasks[task_id] = {'status': 'queued'}
    
    thread = ResearchThread(task_id, question)
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status_url': f'/api/research/{task_id}'
    })

@app.route('/api/research/<task_id>')
def get_research_status(task_id):
    task = research_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    response = {
        'status': task['status'],
        'progress': task.get('progress'),
        'messages': task.get('messages', []),
        'report': task.get('report')
    }
    
    if task['status'] == 'error':
        response['error'] = task.get('error')
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)