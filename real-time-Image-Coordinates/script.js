var tooltip = $( '<div id="tooltip">' ).appendTo( 'body' )[0];

$( '.coords' ).
    each(function () {
        var pos = $( this ).position(),
            top = pos.top,
            left = pos.left,
            width = $( this ).width(),
            height = $( this ).height();
        
        $( this ).
            mousemove(function ( e ) {
                var x = ( ( e.clientX - left ) / width ).toFixed( 1 ),
                    y = ( ( height - ( e.clientY - top ) ) / height ).toFixed( 1 );
                
                $( tooltip ).text( x + ', ' + y ).css({
                    left: e.clientX - 30,
                    top: e.clientY - 30
                }).show();
            }).
            mouseleave(function () {
                $( tooltip ).hide();
            }); 
    });
    



