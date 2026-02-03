"use client";
import { API_BASE_URL } from '../config';
import React, { useState, useEffect } from 'react';
import { Eye, RefreshCw, Layers, Database, MessageSquare, BrainCircuit, Activity, BarChart2, DollarSign, Globe, Shield, TrendingUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const AgentTab = ({ label, icon: Icon, active, onClick }: any) => (
    <button
        onClick={onClick}
        className={`flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-t-md transition-colors ${active
            ? 'bg-[#252525] text-cyan-400 border-t border-x border-[#333]'
            : 'text-gray-500 hover:text-gray-300 hover:bg-[#202020]'
            }`}
    >
        <Icon size={12} />
        {label}
    </button>
);

const AnalystPanel: React.FC = () => {
    const [trace, setTrace] = useState<any>(null);
    const [finalReport, setFinalReport] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<string>('global_analyst');

    // Polling for latest trace
    const fetchLatestTrace = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/pipeline/latest`);
            const data = await res.json();
            if (data.trace) {
                setTrace(data.trace);
                if (data.trace.global_analyst?.output) {
                    setFinalReport(data.trace.global_analyst.output);
                }
            }
        } catch (e) {
            console.error("Failed to fetch pipeline trace", e);
        }
    };

    const triggerPipeline = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/api/pipeline/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: "NIFTY" }) // Default for now
            });
            const data = await res.json();

            if (data.trace) {
                setTrace(data.trace);
                setFinalReport(data.final_report);
                setActiveTab('global_analyst');
            }
        } catch (e) {
            console.error("Failed to trigger pipeline", e);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchLatestTrace();
        // Optional: Poll sparingly if running in background
        // const interval = setInterval(fetchLatestTrace, 10000);
        // return () => clearInterval(interval);
    }, []);

    const renderAgentContent = () => {
        if (!trace || !activeTab) return <div className="p-4 text-gray-500 text-sm">No analysis data available. Run the pipeline.</div>;

        const agentData = trace[activeTab];
        if (!agentData) return <div className="p-4 text-gray-500 text-sm">No data for this agent yet.</div>;

        return (
            <div className="flex flex-col h-full overflow-hidden">
                {/* Agent Header */}
                <div className="p-3 bg-[#202020] border-b border-[#333] flex justify-between items-center">
                    <span className="text-xs font-mono text-gray-400 uppercase">AGENT: {activeTab}</span>
                    {agentData.error ? (
                        <span className="text-xs text-red-400 flex items-center gap-1"><Shield size={10} /> Error</span>
                    ) : (
                        <span className="text-xs text-green-400 flex items-center gap-1"><BrainCircuit size={10} /> Completed</span>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {/* INPUT SECTION */}
                    <div className="space-y-2">
                        <h4 className="text-xs font-bold text-gray-500 flex items-center gap-2">
                            <Database size={12} /> INPUT CONTEXT
                        </h4>
                        <div className="bg-[#111] p-3 rounded border border-[#222] text-xs font-mono text-gray-400 overflow-x-auto max-h-[200px]">
                            <pre>{JSON.stringify(agentData.inputs, null, 2)}</pre>
                        </div>
                    </div>

                    {/* OUTPUT SECTION */}
                    <div className="space-y-2">
                        <h4 className="text-xs font-bold text-cyan-500 flex items-center gap-2">
                            <MessageSquare size={12} /> AGENT RESPONSE
                        </h4>
                        {typeof agentData.output === 'string' ? (
                            <div className="bg-[#1a1a2e] p-3 rounded border border-cyan-900/30 text-sm text-gray-200">
                                <ReactMarkdown>{agentData.output}</ReactMarkdown>
                            </div>
                        ) : (
                            <div className="bg-[#1a1a2e] p-3 rounded border border-cyan-900/30 text-xs font-mono text-cyan-100 overflow-x-auto">
                                <pre>{JSON.stringify(agentData.output, null, 2)}</pre>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col w-full h-full bg-[#151515] border border-gray-800 rounded-lg overflow-hidden">
            {/* Main Header */}
            <div className="p-3 bg-[#2d2d2d] border-b border-gray-700 flex justify-between items-center shrink-0">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <Layers size={16} className="text-cyan-400" /> Commitee of Experts (10-Agent Pipeline)
                </h3>
                <button
                    onClick={triggerPipeline}
                    disabled={isLoading}
                    className="p-1.5 text-xs bg-cyan-900/30 text-cyan-400 border border-cyan-800 rounded hover:bg-cyan-900/50 flex items-center gap-1 transition-colors disabled:opacity-50"
                >
                    <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
                    {isLoading ? "Running Committee..." : "Run Analysis"}
                </button>
            </div>

            {/* Tabs Scroll Area */}
            <div className="flex bg-[#1a1a1a] border-b border-[#333] shrink-0 overflow-x-auto no-scrollbar pt-2 px-2 gap-1">
                <AgentTab label="Global Analyst" icon={BrainCircuit} active={activeTab === 'global_analyst'} onClick={() => setActiveTab('global_analyst')} />
                <AgentTab label="Fundamentals" icon={Globe} active={activeTab === 'fundamentals'} onClick={() => setActiveTab('fundamentals')} />
                <AgentTab label="Technicals" icon={BarChart2} active={activeTab === 'technicals'} onClick={() => setActiveTab('technicals')} />
                <AgentTab label="News" icon={Globe} active={activeTab === 'news'} onClick={() => setActiveTab('news')} />
                <AgentTab label="Sentiment" icon={MessageSquare} active={activeTab === 'sentiment'} onClick={() => setActiveTab('sentiment')} />
                <AgentTab label="Institutional" icon={DollarSign} active={activeTab === 'institutional'} onClick={() => setActiveTab('institutional')} />
                <AgentTab label="Options" icon={Activity} active={activeTab === 'options'} onClick={() => setActiveTab('options')} />
                <AgentTab label="Vol Surface" icon={Activity} active={activeTab === 'vol_surface'} onClick={() => setActiveTab('vol_surface')} />
                <AgentTab label="Liquidity" icon={Database} active={activeTab === 'liquidity'} onClick={() => setActiveTab('liquidity')} />
                <AgentTab label="Correlation" icon={Layers} active={activeTab === 'correlation'} onClick={() => setActiveTab('correlation')} />
                <AgentTab label="Patterns" icon={TrendingUp} active={activeTab === 'chart_pattern'} onClick={() => setActiveTab('chart_pattern')} />
            </div>

            {/* Content Area */}
            <div className="flex-1 min-h-0 relative">
                {renderAgentContent()}
            </div>
        </div>
    );
};

export default AnalystPanel;
