using UnrealBuildTool;

public class PuzzleSequencerEditor : ModuleRules
{
    public PuzzleSequencerEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PublicDependencyModuleNames.AddRange(new string[] {"PuzzleSequencer"});
        
        PrivateDependencyModuleNames.AddRange(new string[] {"Core", "Slate", "SlateCore"});
    }
}