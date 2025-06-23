<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { TheChessboard } from 'vue3-chessboard'
import 'vue3-chessboard/style.css'
import { Chess } from 'chess.js'

const botDepth = 2
const moveDelay = 1000

const board = ref(new Chess())
let boardAPI
const isCalculating = ref(false)
let worker = null

function playNextMove(move) {
  console.log(boardAPI.getTurnColor())
  if (isCalculating.value) {
    return
  }

  const currentFen = boardAPI.getFen()
  board.value.load(currentFen)

  if (board.value.isGameOver()) {
    return
  }

  isCalculating.value = true
  worker.postMessage({ fen: currentFen, depth: botDepth })
}

onMounted(() => {
  worker = new Worker(new URL('./worker', import.meta.url), { type: 'module' })

  worker.onmessage = (event) => {
    isCalculating.value = false
    if (event.data.error) {
      return
    }

    const { move } = event.data
    boardAPI.move(move)
  }

  worker.onerror = (error) => {
    isCalculating.value = false
  }
})

onUnmounted(() => {
  if (worker) {
    worker.terminate()
    worker = null
  }
})
function handleBoardCreatd(api) {
  boardAPI = api
}
</script>

<template>
  <TheChessboard @board-created="handleBoardCreatd" @move="playNextMove" />
</template>
