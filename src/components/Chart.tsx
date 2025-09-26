"use client";

import React from "react";
import {
  LineChart as RCLineChart,
  Line as RCLine,
  XAxis as RCXAxis,
  YAxis as RCYAxis,
  CartesianGrid as RCCartesianGrid,
  Tooltip as RCTooltip,
} from "recharts";

type ChartPoint = { name: string; price: number };

export function Chart({
  data,
  isPositive,
}: {
  data: ChartPoint[];
  isPositive: boolean;
}) {
  return (
    <div className="h-48">
      <RCLineChart width={300} height={180} data={data}>
        <RCCartesianGrid strokeDasharray="3 3" opacity={0.2} />
        <RCXAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#888888" />
        <RCYAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 12 }}
          stroke="#888888"
        />
        <RCTooltip
          contentStyle={{
            backgroundColor: "white",
            border: "1px solid #e5e7eb",
          }}
        />
        <RCLine
          type="monotone"
          dataKey="price"
          stroke={isPositive ? "#10B981" : "#EF4444"}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </RCLineChart>
    </div>
  );
}

export function LargeChart({
  data,
  isPositive,
}: {
  data: ChartPoint[];
  isPositive: boolean;
}) {
  return (
    <div className="h-64">
      <RCLineChart width={500} height={240} data={data}>
        <RCCartesianGrid strokeDasharray="3 3" opacity={0.2} />
        <RCXAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#888888" />
        <RCYAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 12 }}
          stroke="#888888"
        />
        <RCTooltip
          contentStyle={{
            backgroundColor: "white",
            border: "1px solid #e5e7eb",
          }}
        />
        <RCLine
          type="monotone"
          dataKey="price"
          stroke={isPositive ? "#10B981" : "#EF4444"}
          strokeWidth={3}
          dot={false}
          isAnimationActive={false}
        />
      </RCLineChart>
    </div>
  );
}
