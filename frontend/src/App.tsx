import React from 'react';
import ConfigEditor from './components/ConfigEditor';

export default function App() {
  const [section, setSection] = React.useState('home');
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <header className="p-4 border-b border-muted flex items-center justify-between">
        <h1 className="text-xl font-bold">MCP Minimalist Web UI</h1>
        <span className="text-accent">Dark Mode</span>
      </header>
      <main className="flex-1 flex flex-row">
        <aside className="w-64 bg-muted p-4 hidden md:block">
          {/* Navigation or engine list here */}
          <div className="font-semibold mb-2">Navigation</div>
          <ul>
            <li>
              <button className={`w-full text-left py-1 px-2 rounded ${section === 'config' ? 'bg-accent text-white' : ''}`} onClick={() => setSection('config')}>Config</button>
            </li>
            <li>
              <button className={`w-full text-left py-1 px-2 rounded ${section === 'engines' ? 'bg-accent text-white' : ''}`} onClick={() => setSection('engines')}>Engines</button>
            </li>
            <li>
              <button className={`w-full text-left py-1 px-2 rounded ${section === 'knowledgebase' ? 'bg-accent text-white' : ''}`} onClick={() => setSection('knowledgebase')}>Knowledgebase</button>
            </li>
          </ul>
        </aside>
        <section className="flex-1 p-8">
          {section === 'config' && <ConfigEditor />}
          {section === 'home' && (
            <>
              <h2 className="text-lg font-semibold mb-4">Welcome to the MCP Web UI</h2>
              <p>This is a minimalist, dark-themed interface for managing the MCP server.</p>
              <ul className="mt-4 list-disc list-inside text-muted">
                <li>Edit <b>config.cfg</b></li>
                <li>Monitor and control engines/lobes</li>
                <li>Browse and search the knowledgebase</li>
              </ul>
            </>
          )}
          {/* Placeholders for other sections */}
          {section === 'engines' && <div>Engine controls coming soon...</div>}
          {section === 'knowledgebase' && <div>Knowledgebase browser coming soon...</div>}
        </section>
      </main>
      <footer className="p-4 border-t border-muted text-center text-xs text-muted">
        &copy; {new Date().getFullYear()} MCP Server | Minimalist Dark UI
      </footer>
    </div>
  );
} 