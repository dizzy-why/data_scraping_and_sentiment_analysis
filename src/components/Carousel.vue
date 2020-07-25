<template>
  <v-carousel
    hide-delimiter-background
    show-arrows-on-hover
    height="auto"
  >
    <v-carousel-item
      v-for="(item,i) in items"
      :key="i"
      :src="item.src"
      reverse-transition="fade-transition"
      transition="fade-transition"
    ></v-carousel-item>
  </v-carousel>
</template>
<script>
 
  export default {
    name:'Carousel',
    props:[
      'items'
    ],
    
  }
</script>