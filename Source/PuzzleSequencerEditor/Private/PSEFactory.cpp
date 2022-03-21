#include "PuzzleSequencerEditor/Public/PSEFactory.h"

#include "ClassViewerFilter.h"
#include "ClassViewerModule.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "Kismet2/SClassPickerDialog.h"

#define LOCTEXT_NAMESPACE "PuzzleSequencerEditorFactory"

#pragma region ClassFilter
class FAssetClassParentFilter : public IClassViewerFilter
{
public:
	TSet<const UClass*> AllowedChildrenOfClasses{};
	EClassFlags DisallowedClassFlags{CLASS_None};

	bool bDisallowBlueprintBase{false};

	virtual bool IsClassAllowed(const FClassViewerInitializationOptions& InInitOptions, const UClass* InClass, TSharedRef<FClassViewerFilterFuncs> InFilterFuncs) override
	{
		const bool bAllowed = !InClass->HasAnyClassFlags(DisallowedClassFlags) && InFilterFuncs->IfInChildOfClassesSet(AllowedChildrenOfClasses, InClass) != EFilterReturn::Failed;
		if (bAllowed && bDisallowBlueprintBase)
		{
			if (FKismetEditorUtilities::CanCreateBlueprintOfClass(InClass))
			{
				return false;
			}
		}

		return bAllowed;
	}

	virtual bool IsUnloadedClassAllowed(const FClassViewerInitializationOptions& InInitOptions, const TSharedRef<const IUnloadedBlueprintData> InUnloadedClassData, TSharedRef<FClassViewerFilterFuncs> InFilterFuncs) override
	{
		if (bDisallowBlueprintBase)
		{
			return false;
		}

		return !InUnloadedClassData->HasAnyClassFlags(DisallowedClassFlags) && InFilterFuncs->IfInChildOfClassesSet(AllowedChildrenOfClasses, InUnloadedClassData) != EFilterReturn::Failed;
	}
};
#pragma endregion ClassFilter

UPSEFactory::UPSEFactory()
{
	bCreateNew = true;
	bEditAfterNew = true;
	SupportedClass = UPuzzleSequencer::StaticClass();
}

bool UPSEFactory::ConfigureProperties()
{
	PuzzleSequencerClass = nullptr;

	const FClassViewerModule& classViewer = FModuleManager::LoadModuleChecked<FClassViewerModule>("ClassViewer");

	FClassViewerInitializationOptions options;
	options.Mode = EClassViewerMode::ClassPicker;

	const TSharedPtr<FAssetClassParentFilter> filter = MakeShareable(new FAssetClassParentFilter);
	options.ClassFilters.Add(filter.ToSharedRef());

	filter->DisallowedClassFlags = CLASS_Abstract | CLASS_Deprecated | CLASS_NewerVersionExists | CLASS_HideDropDown;
	filter->AllowedChildrenOfClasses.Add(UPuzzleSequencer::StaticClass());

	const FText title = LOCTEXT("CreatePuzzleSequencerAssetOptions", "Pick Puzzle Sequencer Class");
	UClass* chosenClass = nullptr;

	const bool bPressedOk = SClassPickerDialog::PickClass(title, options, chosenClass, UPuzzleSequencer::StaticClass());
	if (bPressedOk)
	{
		PuzzleSequencerClass = chosenClass;
	}

	return bPressedOk;
}

UObject* UPSEFactory::FactoryCreateNew(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn)
{
	if (!IsValid(PuzzleSequencerClass))
	{
		return NewObject<UPuzzleSequencer>(InParent, PuzzleSequencerClass, InName, Flags | RF_Transactional);
	}
	else
	{
		check(InClass->IsChildOf(UPuzzleSequencer::StaticClass()));
		return NewObject<UObject>(InParent, InClass, InName, Flags | RF_Transactional);
	}
}

#undef LOCTEXT_NAMESPACE
