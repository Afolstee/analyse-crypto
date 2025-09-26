import { NextResponse } from "next/server";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const page = searchParams.get("page") || 1;
  const per_page = searchParams.get("per_page") || 10;

  try {
    const response = await fetch(
      `https://analyse-crypto.onrender.com/news?page=${page}&per_page=${per_page}`
    );
    const data = await response.json();

    return new NextResponse(JSON.stringify(data), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, s-maxage=600, stale-while-revalidate=3600",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Failed to fetch news" },
      { status: 500 }
    );
  }
}
