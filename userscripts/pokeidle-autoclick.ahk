; Pokeidle - Auto clique por posição da tela (AutoHotkey v2, Windows)
;   F8  = salva a posição atual do mouse como alvo
;   F9  = liga/desliga os cliques
;   F10 = fecha o script
#Requires AutoHotkey v2.0
#SingleInstance Force
CoordMode "Mouse", "Screen"

intervalo := 1500   ; ms entre cliques
alvoX := 0, alvoY := 0
ligado := false

F8:: {
    global alvoX, alvoY
    MouseGetPos &alvoX, &alvoY
    ToolTip "Alvo salvo: " alvoX ", " alvoY
    SetTimer () => ToolTip(), -1500
}

F9:: {
    global ligado
    if (alvoX = 0 && alvoY = 0) {
        ToolTip "Primeiro coloque o mouse no item e aperte F8"
        SetTimer () => ToolTip(), -2000
        return
    }
    ligado := !ligado
    SetTimer Clicar, ligado ? intervalo : 0
    ToolTip ligado ? "Auto clique LIGADO" : "Auto clique DESLIGADO"
    SetTimer () => ToolTip(), -1500
}

F10::ExitApp

Clicar() {
    ; Clica no alvo e devolve o mouse para onde estava.
    MouseGetPos &x, &y
    Click alvoX, alvoY
    MouseMove x, y, 0
}
