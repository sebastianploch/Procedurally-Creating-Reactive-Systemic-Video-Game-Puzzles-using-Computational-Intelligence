#pragma once

#include "CoreMinimal.h"
#include "AssetTypeActions_Base.h"

class PUZZLESEQUENCEREDITOR_API FAssetTypeActions_PSE : public FAssetTypeActions_Base
{
public:
	FAssetTypeActions_PSE(EAssetTypeCategories::Type InAssetCategory);

	virtual FText GetName() const override;
	virtual FColor GetTypeColor() const override;
	virtual UClass* GetSupportedClass() const override;
	virtual void OpenAssetEditor(const TArray<UObject*>& InObjects, TSharedPtr<IToolkitHost> EditWithinLevelEditor) override;
	virtual uint32 GetCategories() override;

private:
	EAssetTypeCategories::Type AssetCategory{EAssetTypeCategories::None};
};
