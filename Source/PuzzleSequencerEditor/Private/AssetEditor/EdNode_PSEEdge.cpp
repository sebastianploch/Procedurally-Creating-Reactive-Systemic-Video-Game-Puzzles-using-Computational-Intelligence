#include "PuzzleSequencerEditor/Public/AssetEditor/EdNode_PSEEdge.h"
#include "EdGraph/EdGraph.h"
#include "PuzzleSequencerEdge.h"
#include "PuzzleSequencerNode.h"
#include "AssetEditor/EdNode_PSENode.h"

#define LOCTEXT_NAMESPACE "EdNode_PSEEdge"

UEdNode_PSEEdge::UEdNode_PSEEdge()
{
	bCanRenameNode = true;
}

void UEdNode_PSEEdge::SetEdge(UPuzzleSequencerEdge* InEdge)
{
	Edge = InEdge;
}

void UEdNode_PSEEdge::AllocateDefaultPins()
{
	UEdGraphPin* inputs = CreatePin(EGPD_Input, TEXT("Edge"), FName(), TEXT("In"));
	inputs->bHidden = true;
	UEdGraphPin* outputs = CreatePin(EGPD_Output, TEXT("Edge"), FName(), TEXT("Out"));
	outputs->bHidden = true;
}

FText UEdNode_PSEEdge::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
	if (Edge)
	{
		return Edge->GetNodeTitle();
	}
	return FText();
}

void UEdNode_PSEEdge::PinConnectionListChanged(UEdGraphPin* Pin)
{
	if (Pin->LinkedTo.IsEmpty())
	{
		Modify();

		if (UEdGraph* parentGraph = GetGraph())
		{
			parentGraph->Modify();
		}

		DestroyNode();
	}
}

void UEdNode_PSEEdge::PrepareForCopying()
{
	Edge->Rename(nullptr, this, REN_DontCreateRedirectors | REN_DoNotDirty);
}

void UEdNode_PSEEdge::CreateConnections(UEdNode_PSENode* InStart, UEdNode_PSENode* InEnd)
{
	Pins[0]->Modify();
	Pins[0]->LinkedTo.Empty();

	InStart->GetOutputPin()->Modify();
	Pins[0]->MakeLinkTo(InStart->GetOutputPin());

	Pins[1]->Modify();
	Pins[1]->LinkedTo.Empty();

	InEnd->GetInputPin()->Modify();
	Pins[1]->MakeLinkTo(InEnd->GetInputPin());
}

#undef LOCTEXT_NAMESPACE
