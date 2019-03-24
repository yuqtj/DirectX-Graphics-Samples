//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard 
//

#include "stdafx.h"
#include "GameInput.h"
#include "PerformanceTimers.h"
#include "EngineTuning.h"
#include "GpuTimeManager.h"
#include <vector>
#include <unordered_map>
#include <array>

using namespace std;
using namespace DX;

namespace EngineProfiling
{
    bool Paused = false;
}

namespace
{
    wstring Indent(UINT spaces)
    {
        wstring s;
        return s.append(spaces, L' ');
    }
}

// ToDo dedupe with the other GpuTimer
class GpuTimer
{
public:

    GpuTimer::GpuTimer()
    {
        m_TimerIndex = GpuTimeManager::instance().NewTimer();
    }

    void Start(ID3D12GraphicsCommandList* CommandList)
    {
        GpuTimeManager::instance().Start(CommandList, m_TimerIndex);
    }

    void Stop(ID3D12GraphicsCommandList* CommandList)
    {
        GpuTimeManager::instance().Stop(CommandList, m_TimerIndex);
    }

    float GpuTimer::GetElapsedMS(void)
    {
        return GpuTimeManager::instance().GetElapsedMS(m_TimerIndex);
    }

    float GetAverageMS() const
    {
        return GpuTimeManager::instance().GetAverageMS(m_TimerIndex);
    }

    uint32_t GetTimerIndex(void)
    {
        return m_TimerIndex;
    }
private:

    uint32_t m_TimerIndex;
};

class NestedTimingTree
{
public:
    NestedTimingTree(const wstring& name, NestedTimingTree* parent = nullptr)
        : m_Name(name), m_Parent(parent), m_IsExpanded(false) {}

    NestedTimingTree* GetChild(const wstring& name)
    {
        auto iter = m_LUT.find(name);
        if (iter != m_LUT.end())
            return iter->second;

        NestedTimingTree* node = new NestedTimingTree(name, this);
        m_Children.push_back(node);
        m_LUT[name] = node;
        return node;
    }

    NestedTimingTree* NextScope(void)
    {
        if (m_IsExpanded && m_Children.size() > 0)
            return m_Children[0];

        return m_Parent->NextChild(this);
    }

    NestedTimingTree* PrevScope(void)
    {
        NestedTimingTree* prev = m_Parent->PrevChild(this);
        return prev == m_Parent ? prev : prev->LastChild();
    }

    NestedTimingTree* FirstChild(void)
    {
        return m_Children.size() == 0 ? nullptr : m_Children[0];
    }

    NestedTimingTree* LastChild(void)
    {
        if (!m_IsExpanded || m_Children.size() == 0)
            return this;

        return m_Children.back()->LastChild();
    }

    NestedTimingTree* NextChild(NestedTimingTree* curChild)
    {
        assert(curChild->m_Parent == this);

        for (auto iter = m_Children.begin(); iter != m_Children.end(); ++iter)
        {
            if (*iter == curChild)
            {
                auto nextChild = iter; ++nextChild;
                if (nextChild != m_Children.end())
                    return *nextChild;
            }
        }

        if (m_Parent != nullptr)
            return m_Parent->NextChild(this);
        else
            return &sm_RootScope;
    }

    NestedTimingTree* PrevChild(NestedTimingTree* curChild)
    {
        assert(curChild->m_Parent == this);

        if (*m_Children.begin() == curChild)
        {
            if (this == &sm_RootScope)
                return sm_RootScope.LastChild();
            else
                return this;
        }

        for (auto iter = m_Children.begin(); iter != m_Children.end(); ++iter)
        {
            if (*iter == curChild)
            {
                auto prevChild = iter; --prevChild;
                return *prevChild;
            }
        }

        return nullptr;
    }

    void StartTiming(ID3D12GraphicsCommandList* CommandList)
    {
        m_CpuTimer.Start();
        if (CommandList == nullptr)
            return;

        m_GpuTimer.Start(CommandList);

        PIXBeginEvent(CommandList, 0, m_Name.c_str());
    }

    void StopTiming(ID3D12GraphicsCommandList* CommandList)
    {
        m_CpuTimer.Stop();
        if (CommandList == nullptr)
            return;

        m_GpuTimer.Stop(CommandList);

        PIXEndEvent(CommandList);
    }

    void GatherTimes(uint32_t FrameIndex)
    {
        if (EngineProfiling::Paused)
        {
            for (auto node : m_Children)
                node->GatherTimes(FrameIndex);
            return;
        }

        for (auto node : m_Children)
            node->GatherTimes(FrameIndex);

        m_StartTick = 0;
        m_EndTick = 0;
    }

    static void PushProfilingMarker(const wstring& name, ID3D12GraphicsCommandList* CommandList);
    static void PopProfilingMarker(ID3D12GraphicsCommandList* CommandList);
    static void Update(void);

    float GetAverageCpuTimeMS(void) const { return m_GpuTimer.GetAverageMS(); }
    float GetAverageGpuTimeMS(void) const { return m_CpuTimer.GetAverageMS(); }


    static void Display(wstringstream& Text, UINT indent)
    {
        sm_RootScope.DisplayNode(Text, indent);
    }

    static const NestedTimingTree& Root() { return sm_RootScope; }
private:

    void DisplayNode(wstringstream& Text, UINT indent);
    void StoreToGraph(void);
    void DeleteChildren(void)
    {
        for (auto node : m_Children)
            delete node;
        m_Children.clear();
    }

    wstring m_Name;
    NestedTimingTree* m_Parent;
    vector<NestedTimingTree*> m_Children;
    unordered_map<wstring, NestedTimingTree*> m_LUT;
    int64_t m_StartTick;
    int64_t m_EndTick;
    bool m_IsExpanded;
    CPUTimer m_CpuTimer;
    GpuTimer m_GpuTimer;
    static NestedTimingTree sm_RootScope;
    static NestedTimingTree* sm_CurrentNode;
    static NestedTimingTree* sm_SelectedScope;

};

NestedTimingTree NestedTimingTree::sm_RootScope(L"");
NestedTimingTree* NestedTimingTree::sm_CurrentNode = &NestedTimingTree::sm_RootScope;
NestedTimingTree* NestedTimingTree::sm_SelectedScope = &NestedTimingTree::sm_RootScope;
namespace EngineProfiling
{
    BoolVar DrawFrameRate(L"Display Frame Rate", true);
    BoolVar DrawProfiler(L"Display Profiler", false);

    void Update(void)
    {
        if (GameInput::IsFirstPressed(GameInput::kStartButton)
            || GameInput::IsFirstPressed(GameInput::kKey_space))
        {
            Paused = !Paused;
        }
    }
    
    void BeginFrame(ID3D12GraphicsCommandList* CommandList)
    {
        GpuTimeManager::instance().BeginFrame(CommandList);
    }

    void EndFrame(ID3D12GraphicsCommandList* CommandList)
    {
        GpuTimeManager::instance().EndFrame(CommandList);
    }

    void RestoreDevice(ID3D12Device* Device, ID3D12CommandQueue* CommandQueue, UINT MaxFrameCount, UINT MaxNumTimers)
    {
        GpuTimeManager::instance().RestoreDevice(Device, CommandQueue, MaxFrameCount, MaxNumTimers);
    }

    void ReleaseDevice()
    {
        GpuTimeManager::instance().ReleaseDevice();
    }

    void BeginBlock(const wstring& name, ID3D12GraphicsCommandList* CommandList)
    {
        NestedTimingTree::PushProfilingMarker(name, CommandList);
    }

    void EndBlock(ID3D12GraphicsCommandList* CommandList)
    {
        NestedTimingTree::PopProfilingMarker(CommandList);
    }

    bool IsPaused()
    {
        return Paused;
    }

    void DisplayFrameRate(wstringstream& Text, UINT indent)
    {
        if (!DrawFrameRate)
            return;

        float cpuTime = NestedTimingTree::Root().GetAverageCpuTimeMS();
        float gpuTime = NestedTimingTree::Root().GetAverageGpuTimeMS();
        float frameRate = 1e3f / cpuTime;

        streamsize prevPrecision = Text.precision(3);
        //streamsize prevWidth = Text.width(7);
        Text << Indent(indent)
             << L"CPU " << cpuTime << L"ms, "
             << L"GPU " << gpuTime << L"ms, ";
        Text.width(3);
        Text << (uint32_t)(frameRate + 0.5f) << L" FPS\n";

       // Text.width(prevWidth);
        Text.precision(prevPrecision);
    }

    void Display(wstringstream& Text, UINT indent)
    {
        if (DrawProfiler)
        {
            NestedTimingTree::Update();

            Text << Indent(indent) << L"Engine Profiling (use arrow keys) "
                 << L"           CPU [ms]    GPU [ms]\n";

            NestedTimingTree::Display(Text, indent);
        }
    }

} // EngineProfiling

void NestedTimingTree::PushProfilingMarker(const wstring& name, ID3D12GraphicsCommandList* CommandList)
{
    sm_CurrentNode = sm_CurrentNode->GetChild(name);
    sm_CurrentNode->StartTiming(CommandList);
}

void NestedTimingTree::PopProfilingMarker(ID3D12GraphicsCommandList* CommandList)
{
    sm_CurrentNode->StopTiming(CommandList);
    sm_CurrentNode = sm_CurrentNode->m_Parent;
}

void NestedTimingTree::Update(void)
{
    assert(sm_SelectedScope != nullptr && L"Corrupted profiling data structure");

    if (sm_SelectedScope == &sm_RootScope)
    {
        sm_SelectedScope = sm_RootScope.FirstChild();
        if (sm_SelectedScope == &sm_RootScope)
            return;
    }

    if (GameInput::IsFirstPressed(GameInput::kDPadLeft)
        || GameInput::IsFirstPressed(GameInput::kKey_left))
    {
        sm_SelectedScope->m_IsExpanded = false;
    }
    else if (GameInput::IsFirstPressed(GameInput::kDPadRight)
        || GameInput::IsFirstPressed(GameInput::kKey_right))
    {
        sm_SelectedScope->m_IsExpanded = true;
    }
    else if (GameInput::IsFirstPressed(GameInput::kDPadDown)
        || GameInput::IsFirstPressed(GameInput::kKey_down))
    {
        sm_SelectedScope = sm_SelectedScope ? sm_SelectedScope->NextScope() : nullptr;
    }
    else if (GameInput::IsFirstPressed(GameInput::kDPadUp)
        || GameInput::IsFirstPressed(GameInput::kKey_up))
    {
        sm_SelectedScope = sm_SelectedScope ? sm_SelectedScope->PrevScope() : nullptr;
    }
}

void NestedTimingTree::DisplayNode(wstringstream& Text, UINT indent)
{

    if (this == &sm_RootScope)
    {
        m_IsExpanded = true;
        sm_RootScope.FirstChild()->m_IsExpanded = true;
    }
    else
    {
        Text << (sm_SelectedScope == this ? L"[x] " : L"[] ");

        Text << Indent(indent);

        if (m_Children.size() == 0)
            Text << L"   ";
        else if (m_IsExpanded)
            Text << L"-  ";
        else
            Text << L"+  ";

        Text << m_Name.c_str();

        streamsize prevPrecision = Text.precision(3);
        streamsize prevWidth = Text.width(6);
        Text << m_CpuTimer.GetAverageMS() << L" "
             << m_GpuTimer.GetAverageMS() << L"   ";
        Text.width(prevWidth);
        Text.precision(prevPrecision);

        Text << L"\n";
    }

    if (!m_IsExpanded)
        return;

    for (auto node : m_Children)
        node->DisplayNode(Text, indent + 2);
}

