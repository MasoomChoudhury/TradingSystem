"use client";

import { useState, useEffect } from "react";
import { Activity, CheckCircle, XCircle, Clock, Loader2, ChevronDown, ChevronUp } from "lucide-react";

interface TaskStep {
    name: string;
    status: string;
    result?: string;
}

interface Task {
    task_id: string;
    name: string;
    description?: string;
    state: string;
    progress: number;
    current_step: string;
    started_at: string;
    updated_at: string;
    steps?: TaskStep[];
}

export default function TaskStatusPanel() {
    const [activeTasks, setActiveTasks] = useState<Task[]>([]);
    const [recentTasks, setRecentTasks] = useState<Task[]>([]);
    const [isExpanded, setIsExpanded] = useState(true);
    const [showRecent, setShowRecent] = useState(false);

    useEffect(() => {
        // Initial fetch
        fetchTasks();

        // Poll every 2 seconds for active tasks
        const interval = setInterval(fetchTasks, 2000);
        return () => clearInterval(interval);
    }, []);

    const fetchTasks = async () => {
        try {
            const [activeRes, recentRes] = await Promise.all([
                fetch("http://127.0.0.1:8000/api/tasks/active"),
                fetch("http://127.0.0.1:8000/api/tasks/recent?limit=5")
            ]);

            if (activeRes.ok) {
                const activeData = await activeRes.json();
                setActiveTasks(activeData.tasks || []);
                // Auto-expand if there are active tasks
                if (activeData.tasks?.length > 0) {
                    setIsExpanded(true);
                }
            }

            if (recentRes.ok) {
                const recentData = await recentRes.json();
                setRecentTasks(recentData.tasks || []);
            }
        } catch (e) {
            console.error("Failed to fetch tasks:", e);
        }
    };

    const getStateIcon = (state: string) => {
        switch (state) {
            case "running":
                return <Loader2 size={14} className="text-blue-400 animate-spin" />;
            case "completed":
                return <CheckCircle size={14} className="text-green-400" />;
            case "failed":
                return <XCircle size={14} className="text-red-400" />;
            default:
                return <Clock size={14} className="text-gray-400" />;
        }
    };

    const getStateColor = (state: string) => {
        switch (state) {
            case "running":
                return "text-blue-400";
            case "completed":
                return "text-green-400";
            case "failed":
                return "text-red-400";
            default:
                return "text-gray-400";
        }
    };

    const formatTime = (isoString: string) => {
        const date = new Date(isoString);
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    const hasContent = activeTasks.length > 0 || (showRecent && recentTasks.length > 0);

    return (
        <div className="bg-zinc-900/50 rounded-lg border border-zinc-800 overflow-hidden">
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full p-3 flex items-center justify-between hover:bg-zinc-800/50 transition-colors"
            >
                <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
                    <Activity size={16} className={activeTasks.length > 0 ? "text-blue-400 animate-pulse" : "text-zinc-500"} />
                    Agent Tasks
                    {activeTasks.length > 0 && (
                        <span className="bg-blue-600 text-white text-[10px] px-1.5 py-0.5 rounded-full">
                            {activeTasks.length} running
                        </span>
                    )}
                </h3>
                {isExpanded ? <ChevronUp size={16} className="text-zinc-500" /> : <ChevronDown size={16} className="text-zinc-500" />}
            </button>

            {/* Content */}
            {isExpanded && (
                <div className="border-t border-zinc-800">
                    {/* Active Tasks */}
                    {activeTasks.length > 0 ? (
                        <div className="p-2 space-y-2">
                            {activeTasks.map((task) => (
                                <div key={task.task_id} className="bg-zinc-800/50 rounded-lg p-3">
                                    {/* Task Header */}
                                    <div className="flex items-center gap-2 mb-2">
                                        {getStateIcon(task.state)}
                                        <span className="text-sm font-medium text-zinc-200">{task.name}</span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="mb-2">
                                        <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                            <span>{task.current_step}</span>
                                            <span>{task.progress}%</span>
                                        </div>
                                        <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-blue-500 transition-all duration-300"
                                                style={{ width: `${task.progress}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Steps */}
                                    {task.steps && task.steps.length > 0 && (
                                        <div className="mt-2 space-y-1">
                                            {task.steps.map((step, idx) => (
                                                <div
                                                    key={idx}
                                                    className={`flex items-center gap-2 text-[10px] ${step.status === "completed"
                                                        ? "text-green-400"
                                                        : step.status === "running"
                                                            ? "text-blue-400"
                                                            : "text-zinc-500"
                                                        }`}
                                                >
                                                    {step.status === "completed" ? (
                                                        <CheckCircle size={10} />
                                                    ) : step.status === "running" ? (
                                                        <Loader2 size={10} className="animate-spin" />
                                                    ) : (
                                                        <Clock size={10} />
                                                    )}
                                                    <span>{step.name}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Started time */}
                                    <div className="mt-2 text-[9px] text-zinc-600">
                                        Started: {formatTime(task.started_at)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="p-4 text-center text-zinc-500 text-xs">
                            No active tasks
                        </div>
                    )}

                    {/* Toggle Recent */}
                    {recentTasks.length > 0 && (
                        <>
                            <button
                                onClick={() => setShowRecent(!showRecent)}
                                className="w-full px-3 py-2 text-[10px] text-zinc-500 hover:text-zinc-400 hover:bg-zinc-800/30 border-t border-zinc-800 flex items-center justify-center gap-1"
                            >
                                {showRecent ? "Hide" : "Show"} recent tasks ({recentTasks.length})
                            </button>

                            {/* Recent Tasks */}
                            {showRecent && (
                                <div className="p-2 space-y-1 border-t border-zinc-800">
                                    {recentTasks.map((task, idx) => (
                                        <div
                                            key={idx}
                                            className="flex items-center gap-2 px-2 py-1.5 rounded text-xs"
                                        >
                                            {getStateIcon(task.state)}
                                            <span className={`flex-1 ${getStateColor(task.state)}`}>
                                                {task.name}
                                            </span>
                                            <span className="text-[9px] text-zinc-600">
                                                {task.progress}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
