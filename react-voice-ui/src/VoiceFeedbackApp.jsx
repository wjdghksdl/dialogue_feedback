import React, { useState, useRef } from "react";
import "./VoiceFeedbackApp.css"; 

// ---------------------------------------------------------
// 결과 대시보드
// ---------------------------------------------------------
const ResultDashboard = ({ data }) => {
  if (!data) return null;

  const scores = data.scores || {};
  const feedback = data.feedback || "피드백 데이터가 없습니다.";

  const metrics = [
    { label: "말하기 속도", score: scores.speed, color: "text-blue-600", bg: "bg-blue-50" },
    { label: "음정/억양", score: scores.pitch, color: "text-purple-600", bg: "bg-purple-50" },
    { label: "대화 습관", score: scores.habit, color: "text-green-600", bg: "bg-green-50" },
    { label: "구조적 안정성", score: scores.structure, color: "text-orange-600", bg: "bg-orange-50" },
    { label: "청자 편의성", score: scores.comfort, color: "text-indigo-600", bg: "bg-indigo-50" },
  ];

  return (
    <div className="bg-white text-gray-800 p-8 rounded-3xl shadow-2xl mt-10 animate-fade-in-up">
      <div className="flex items-center justify-between border-b pb-4 mb-6">
        <h2 className="text-2xl font-bold text-gray-900">📊 종합 대화 능력 평가</h2>
        <div className="flex items-center gap-2">
           <span className="text-sm text-gray-500">종합 점수</span>
           <span className="text-3xl font-black text-gray-800">{scores.overall}점</span>
        </div>
      </div>

      {/* 5대 지표 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        {metrics.map((m, idx) => (
          <div key={idx} className={`${m.bg} p-4 rounded-xl text-center border border-gray-100 shadow-sm`}>
            <p className="text-xs text-gray-500 font-bold mb-1">{m.label}</p>
            <p className={`text-2xl font-extrabold ${m.color}`}>
              {m.score ?? 0}
            </p>
          </div>
        ))}
      </div>

      {/* 텍스트 피드백 */}
      <div className="bg-gray-50 p-6 rounded-2xl border border-gray-200">
        <h3 className="font-bold text-gray-700 mb-4 flex items-center gap-2">
          💡 AI 상세 피드백
        </h3>
        <p className="text-gray-700 whitespace-pre-line leading-7 text-sm">
          {feedback}
        </p>
      </div>
    </div>
  );
};

// ---------------------------------------------------------
// 메인 앱
// ---------------------------------------------------------
export default function VoiceFeedbackApp() {
  const [role, setRole] = useState("일반대화");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await handleUpload(file);
  };

  const handleUpload = async (file) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("role", role);

    try {
      const res = await fetch("http://localhost:8000/upload_audio", {
        method: "POST",
        body: formData,
      });
      const rawData = await res.json();

      if (!res.ok) throw new Error(rawData.error || "서버 에러 발생");

      // 🔥 핵심: 서버 JSON 그대로 사용!
      setResult(rawData);

    } catch (err) {
      alert("오류 발생: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div className="voice-lab-container">

      <div className="hero-section">
        <h1 className="main-title">VOICE LAB</h1>
        <p className="sub-description">
          AI 기반 스피치 분석 & 코칭 솔루션<br />
          당신의 목소리에 숨겨진 잠재력을 발견하세요.
        </p>

        <div className="mb-8">
          <select 
              value={role} 
              onChange={(e) => setRole(e.target.value)}
              className="role-select"
          >
              <option value="일반대화">일상 대화 분석</option>
              <option value="면접">실전 면접 연습</option>
              <option value="발표">프레젠테이션 코칭</option>
          </select>
        </div>
        <br>
        </br>
        <div className="action-buttons">
          <button className="btn btn-record" onClick={() => alert("실시간 녹음 기능은 준비 중입니다!")}>
            <span>🎙️</span> 실시간 녹음
          </button>

          <button 
            className="btn btn-upload" 
            onClick={() => fileInputRef.current.click()} 
            disabled={loading}
          >
            {loading ? "⏳ 분석 중입니다..." : <><span>📁</span> 음성 파일 업로드</>}
          </button>
        </div>

        <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/*"
            style={{ display: "none" }} 
        />
      </div>

      {result && (
        <div className="dashboard-container">
          <ResultDashboard data={result} />
        </div>
      )}
    </div>
  );
}
