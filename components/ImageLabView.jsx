"use client";
 
// 图片识别页。结构照着 TextLabView 写，只是把文字换成了图片。
import { useState } from "react";
import AnimatedCardGrid from "./AnimatedCardGrid.jsx";
import ImageHistoryModal from "./ImageHistoryModal.jsx";
import ImageInputCard from "./ImageInputCard.jsx";
import ImageResultCard from "./ImageResultCard.jsx";
import Nav from "./Nav.jsx";
import PageHeading from "./PageHeading.jsx";
import { imageLab } from "../data/site.js";
 
const API = process.env.NEXT_PUBLIC_API_BASE_URL;
 
export default function ImageLabView() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyOpen, setHistoryOpen] = useState(false);
 
  async function openHistory() {
    setHistoryOpen(true);
    try {
      const res = await fetch(`${API}/api/predictions`, {
        credentials: "include",
      });
      setHistory(await res.json());
    } catch {
      // 后端没启动时，弹窗显示「还没有记录」，页面不崩
    }
  }
 
  return (
    <AnimatedCardGrid className="dashboard-grid">
      <article className="hero-stage panel-full">
        <Nav />
        <PageHeading
          title={imageLab.heroTitle}
          subtitle={imageLab.heroSubtitle}
        />
      </article>
 
      <ImageInputCard onResult={setResult} />
      <ImageResultCard result={result} onOpenHistory={openHistory} />
 
      <ImageHistoryModal
        open={historyOpen}
        items={history}
        onClose={() => setHistoryOpen(false)}
      />
    </AnimatedCardGrid>
  );
}
