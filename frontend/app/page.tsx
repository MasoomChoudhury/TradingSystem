"use client";

import { useState } from "react";
import Console from "./components/Console";
import ChatInterface from "./components/ChatInterface";
import SupervisorPanel from "./components/SupervisorPanel";
import ExecutorPanel from "./components/ExecutorPanel";
import APILogPanel from "./components/APILogPanel";
import OpenAlgoControlPanel from "./components/OpenAlgoControlPanel";
import TaskStatusPanel from "./components/TaskStatusPanel";
import AnalystPanel from "./components/AnalystPanel";
import AgentCommsPanel from "./components/AgentCommsPanel";
import SessionPanel from "./components/SessionPanel";

export default function Home() {
  const [activePanel, setActivePanel] = useState<'orchestrator' | 'analyst' | 'supervisor' | 'executor' | 'comms'>('orchestrator');

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-gray-300 font-sans overflow-hidden">

      {/* LEFT: Agent Panels */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-zinc-800">
        {/* Tab Switcher */}
        <div className="flex bg-[#151515] border-b border-zinc-800 shrink-0">
          <button
            onClick={() => setActivePanel('orchestrator')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activePanel === 'orchestrator'
              ? 'text-blue-400 border-b-2 border-blue-400 bg-[#1a1a1a]'
              : 'text-gray-500 hover:text-gray-300'
              }`}
          >
            🎯 Orchestrator
          </button>
          <button
            onClick={() => setActivePanel('analyst')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activePanel === 'analyst'
              ? 'text-cyan-400 border-b-2 border-cyan-400 bg-[#1a1a1a]'
              : 'text-gray-500 hover:text-gray-300'
              }`}
          >
            👁️ Analyst
          </button>
          <button
            onClick={() => setActivePanel('supervisor')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activePanel === 'supervisor'
              ? 'text-purple-400 border-b-2 border-purple-400 bg-[#1a1a1a]'
              : 'text-gray-500 hover:text-gray-300'
              }`}
          >
            🛡️ Supervisor
          </button>
          <button
            onClick={() => setActivePanel('executor')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activePanel === 'executor'
              ? 'text-amber-400 border-b-2 border-amber-400 bg-[#1a1a1a]'
              : 'text-gray-500 hover:text-gray-300'
              }`}
          >
            ⚡ Executor
          </button>
          <button
            onClick={() => setActivePanel('comms')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activePanel === 'comms'
              ? 'text-green-400 border-b-2 border-green-400 bg-[#1a1a1a]'
              : 'text-gray-500 hover:text-gray-300'
              }`}
          >
            💬 Comms
          </button>
        </div>

        {/* Panel Content - All panels stay mounted to preserve state */}
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col bg-[#0f0f0f] relative">
          <div className={`absolute inset-0 ${activePanel === 'orchestrator' ? 'z-10' : 'z-0 opacity-0 pointer-events-none'}`}>
            <ChatInterface />
          </div>
          <div className={`absolute inset-0 ${activePanel === 'analyst' ? 'z-10' : 'z-0 opacity-0 pointer-events-none'}`}>
            <AnalystPanel />
          </div>
          <div className={`absolute inset-0 ${activePanel === 'supervisor' ? 'z-10' : 'z-0 opacity-0 pointer-events-none'}`}>
            <SupervisorPanel />
          </div>
          <div className={`absolute inset-0 ${activePanel === 'executor' ? 'z-10' : 'z-0 opacity-0 pointer-events-none'}`}>
            <ExecutorPanel />
          </div>
          <div className={`absolute inset-0 ${activePanel === 'comms' ? 'z-10' : 'z-0 opacity-0 pointer-events-none'}`}>
            <AgentCommsPanel />
          </div>
        </div>
      </div>

      {/* RIGHT: Trading Session + Controls + Logs */}
      <div className="w-[400px] flex flex-col shrink-0 bg-[#0a0a0a] overflow-hidden">
        {/* Trading Session Panel */}
        <div className="shrink-0 p-2 border-b border-zinc-800">
          <SessionPanel />
        </div>

        {/* OpenAlgo Control Panel */}
        <div className="shrink-0 p-2 border-b border-zinc-800">
          <OpenAlgoControlPanel />
        </div>

        {/* API Log Panel - Takes remaining space */}
        <div className="flex-1 min-h-0 p-2">
          <APILogPanel />
        </div>
      </div>

    </div>
  );
}
